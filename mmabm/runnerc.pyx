# distutils: language = c++

import random

cimport cython

import numpy as np
cimport numpy as np
import pandas as pd

from mmabm.sharedc cimport Side, OType

cimport mmabm.traderc as trader
cimport mmabm.orderbookc as orderbook


cdef class Runner(object):

    cdef orderbook.Orderbook exchange
    
    cdef trader.InformedTrader informed_trader
    cdef trader.PennyJumper pennyjumper
    
    cdef public str h5filename
    cdef int mpi, write_interval, informedTrades
    cdef int prime1, run_steps
    cdef list providers
    cdef bint provider, taker, informed, pj, marketmaker
    cdef np.ndarray t_delta_p, provider_array, t_delta_t, taker_array, t_delta_m, marketmakers, q_take, lambda_t, takerTradeV
    cdef set t_delta_i
    cdef double q_provide, alpha_pj
    cdef dict liquidity_providers
    
    def __init__(self, h5filename='test.h5', mpi=1, prime1=20, run_steps=100000, write_interval=5000, **kwargs):
        self.exchange = orderbook.Orderbook()
        self.h5filename = h5filename
        self.mpi = mpi
        self.run_steps = run_steps + 1
        self.providers = []
        self.provider = kwargs.pop('Provider')
        if self.provider:
            self.t_delta_p, self.provider_array = self.buildProviders(kwargs['numProviders'], kwargs['providerMaxQ'],
                                                                      kwargs['pAlpha'], kwargs['pDelta'])
            self.q_provide = kwargs['qProvide']
            self.providers.append('Provider')
        self.taker = kwargs.pop('Taker')
        if self.taker:
            self.t_delta_t, self.taker_array = self.buildTakers(kwargs['numTakers'], kwargs['takerMaxQ'], kwargs['tMu'])
        self.informed = kwargs.pop('InformedTrader')
        if self.informed:
            if self.taker:
                takerTradeV = np.array([t.quantity for t in self.taker_array])
            informedTrades = np.int(kwargs['iMu']*np.sum(takerTradeV*self.run_steps/self.t_delta_t) if self.taker else 1/kwargs['iMu'])
            self.t_delta_i, self.informed_trader = self.buildInformedTrader(kwargs['informedMaxQ'], kwargs['informedRunLength'], informedTrades, prime1)
        self.pj = kwargs.pop('PennyJumper')
        if self.pj:
            self.pennyjumper = self.buildPennyJumper()
            self.alpha_pj = kwargs['AlphaPJ']
        self.marketmaker = kwargs.pop('MarketMaker')
        if self.marketmaker:
            self.t_delta_m, self.marketmakers = self.buildMarketMakers(kwargs['MMMaxQ'], kwargs['NumMMs'], kwargs['MMQuotes'], 
                                                                       kwargs['MMQuoteRange'], kwargs['MMDelta'])
            self.providers.append('MarketMaker')
        self.q_take, self.lambda_t = self.makeQTake(kwargs['QTake'], kwargs['Lambda0'], kwargs['WhiteNoise'], kwargs['CLambda'])
        self.liquidity_providers = self.makeLiquidityProviders()
        self.seedOrderbook()
        if self.provider:
            self.makeSetup(prime1, kwargs['Lambda0'])
        if self.pj:
            self.runMcsPJ(prime1, write_interval)
        else:
            self.runMcs(prime1, write_interval)
        self.exchange.trade_book_to_h5(h5filename)
        self.qTakeToh5()
        self.mmProfitabilityToh5()
                  
    def buildProviders(self, numProviders, providerMaxQ, pAlpha, pDelta):
        ''' Providers id starts with 1
        '''
        providers_list = [1000 + i for i in range(numProviders)]
        providers = np.array([trader.Provider(p,providerMaxQ,pDelta) for p in providers_list])
        provider_size = np.array([p.quantity for p in providers])
        t_delta_p = np.floor(np.random.exponential(1/pAlpha, numProviders)+1)*provider_size
        return t_delta_p, providers
    
    def buildTakers(self, numTakers, takerMaxQ, tMu):
        ''' Takers id starts with 2
        '''
        takers_list = [2000 + i for i in range(numTakers)]
        takers = np.array([trader.Taker(t, takerMaxQ) for t in takers_list])
        taker_size = np.array([t.quantity for t in takers])
        t_delta_t = np.floor(np.random.exponential(1/tMu, numTakers)+1)*taker_size
        return t_delta_t, takers
    
    def buildInformedTrader(self, informedMaxQ, informedRunLength, informedTrades, prime1):
        ''' Informed trader id starts with 5
        '''
        informed = trader.InformedTrader(5000, informedMaxQ)
        numChoices = int(informedTrades/(informedRunLength*informed.quantity)) + 1
        choiceRange = range(prime1, self.run_steps - informedRunLength + 1)
        t_delta_i = set()
        for _ in range(1, numChoices):
            runL = 0
            step = random.choice(choiceRange)
            while runL < informedRunLength:
                while step in t_delta_i:
                    step += 1
                t_delta_i.add(step)
                step += 1
                runL += 1
        return t_delta_i, informed
    
    def buildPennyJumper(self):
        ''' PJ id starts with 4
        '''
        return trader.PennyJumper(4000, 1, self.mpi)

    def buildMarketMakers(self, mMMaxQ, numMMs, mMQuotes, mMQuoteRange, mMDelta):
        ''' MM id starts with 3
        '''
        marketmakers_list = [3000 + i for i in range(numMMs)]
        marketmakers = np.array([trader.MarketMaker(p, mMMaxQ, mMDelta, mMQuotes, mMQuoteRange) for p in marketmakers_list])
        t_delta_m = np.array([m.quantity for m in marketmakers])
        return t_delta_m, marketmakers
    
    def makeQTake(self, q_take, lambda_0, wn, c_lambda):
        if q_take:
            noise = np.random.rand(2, self.run_steps)
            qt_take = np.empty_like(noise)
            qt_take[:,0] = 0.5
            for i in range(1, self.run_steps):
                qt_take[:,i] = qt_take[:,i-1] + (noise[:,i-1]>qt_take[:,i-1])*wn - (noise[:,i-1]<qt_take[:,i-1])*wn
            lambda_t = -lambda_0*(1 + (np.abs(qt_take[1] - 0.5)/np.sqrt(np.mean(np.square(qt_take[0] - 0.5))))*c_lambda)
            return qt_take[1], lambda_t
        else:
            qt_take = np.array([0.5]*self.run_steps)
            lambda_t = np.array([-lambda_0]*self.run_steps)
            return qt_take, lambda_t
        
    def makeLiquidityProviders(self):
        lp_dict = {}
        if self.provider:
            temp_dict = dict(zip([x.trader_id for x in self.provider_array], list(self.provider_array)))
            lp_dict.update(temp_dict)
        if self.marketmaker:
            temp_dict = dict(zip([x.trader_id for x in self.marketmakers], list(self.marketmakers)))
            lp_dict.update(temp_dict)
        if self.pj:
            lp_dict.update({self.pennyjumper.trader_id: self.pennyjumper})
        return lp_dict
    
    def seedOrderbook(self):
        seed_provider = trader.Provider(9999, 1, 0.05)
        self.liquidity_providers.update({9999: seed_provider})
        ba = random.choice(range(1000005, 1002001, 5))
        bb = random.choice(range(997995, 999996, 5))
        qask = {'order_id': 1, 'trader_id': 9999, 'timestamp': 0, 'type': OType.ADD, 
                'quantity': 1, 'side': Side.ASK, 'price': ba}
        qbid = {'order_id': 2, 'trader_id': 9999, 'timestamp': 0, 'type': OType.ADD,
                'quantity': 1, 'side': Side.BID, 'price': bb}
        seed_provider.local_book[1] = qask
        self.exchange.add_order_to_book(qask)
        self.exchange.add_order_to_history(qask)
        seed_provider.local_book[2] = qbid
        self.exchange.add_order_to_book(qbid)
        self.exchange.add_order_to_history(qbid)
    
    @cython.boundscheck(False)   
    cdef void makeSetup(self, int prime1, float lambda0):
        cdef int current_time
        cdef trader.Provider p
        top_of_book = self.exchange.report_top_of_book(0)
        for current_time in range(1, prime1):
            for p in self.makeProviders(current_time):
                p.process_signal(current_time, top_of_book, self.q_provide, -lambda0)
                self.exchange.process_order(p.quote_collector[-1])
                top_of_book = self.exchange.report_top_of_book(current_time)
                
    cdef np.ndarray makeProviders(self, int step):
        cdef np.ndarray providers = self.provider_array[np.remainder(step, self.t_delta_p)==0]
        np.random.shuffle(providers)
        return providers
    
    def makeAll(self, step):
        trader_list = []
        if self.provider:
            providers_mask = np.remainder(step, self.t_delta_p)==0
            providers = np.vstack((self.provider_array, providers_mask)).T
            trader_list.append(providers)
        if self.taker:
            takers_mask = np.remainder(step, self.t_delta_t)==0
            if takers_mask.any():
                takers = np.vstack((self.taker_array, takers_mask)).T
                trader_list.append(takers[takers_mask])
        if self.marketmaker:
            marketmakers_mask = np.remainder(step, self.t_delta_m)==0
            marketmakers = np.vstack((self.marketmakers, marketmakers_mask)).T
            trader_list.append(marketmakers)
        if self.informed:
            informed_mask = step in self.t_delta_i
            if informed_mask:
                informed = np.array([[self.informed_trader, informed_mask]])
                trader_list.append(informed)
        all_traders = np.vstack(tuple(trader_list))
        np.random.shuffle(all_traders)
        return all_traders
    
    cdef void doCancels(self, trader):
        cdef dict c
        for c in trader.cancel_collector:
            self.exchange.process_order(c)
                    
    cdef void confirmTrades(self):
        cdef dict c
        for c in self.exchange.confirm_trade_collector:
            contra_side = self.liquidity_providers[c['trader']]
            contra_side.confirm_trade_local(c)
            
    @cython.boundscheck(False)
    def runMcs(self, int prime1, int write_interval):
        cdef int current_time
        cdef np.ndarray row
        cdef dict q
        top_of_book = self.exchange.report_top_of_book(prime1)
        for current_time in range(prime1, self.run_steps):
            for row in self.makeAll(current_time):
                if row[0].trader_type in self.providers:
                    if row[1]:
                        row[0].process_signal(current_time, top_of_book, self.q_provide, self.lambda_t[current_time])
                        for q in row[0].quote_collector:
                            self.exchange.process_order(q)
                        top_of_book = self.exchange.report_top_of_book(current_time)
                    row[0].bulk_cancel(current_time)
                    if row[0].cancel_collector:
                        self.doCancels(row[0])
                        top_of_book = self.exchange.report_top_of_book(current_time)
                else:
                    row[0].process_signal(current_time, self.q_take[current_time])
                    self.exchange.process_order(row[0].quote_collector[-1])
                    if self.exchange.traded:
                        self.confirmTrades()
                        top_of_book = self.exchange.report_top_of_book(current_time)
            if not current_time % write_interval:
                self.exchange.order_history_to_h5(self.h5filename)
                self.exchange.sip_to_h5(self.h5filename)
    
    @cython.boundscheck(False)            
    def runMcsPJ(self, int prime1, int write_interval):
        cdef int current_time
        cdef np.ndarray row
        cdef dict q, c
        top_of_book = self.exchange.report_top_of_book(prime1)
        for current_time in range(prime1, self.run_steps):
            for row in self.makeAll(current_time):
                if row[0].trader_type in self.providers:
                    if row[1]:
                        row[0].process_signal(current_time, top_of_book, self.q_provide, self.lambda_t[current_time])
                        for q in row[0].quote_collector:
                            self.exchange.process_order(q)
                        top_of_book = self.exchange.report_top_of_book(current_time)
                    row[0].bulk_cancel(current_time)
                    if row[0].cancel_collector:
                        self.doCancels(row[0])
                        top_of_book = self.exchange.report_top_of_book(current_time)
                else:
                    row[0].process_signal(current_time, self.q_take[current_time])
                    self.exchange.process_order(row[0].quote_collector[-1])
                    if self.exchange.traded:
                        self.confirmTrades()
                        top_of_book = self.exchange.report_top_of_book(current_time)
                if random.random() < self.alpha_pj:
                    self.pennyjumper.process_signal(current_time, top_of_book, self.q_take[current_time])
                    if self.pennyjumper.cancel_collector:
                        for c in self.pennyjumper.cancel_collector:
                            self.exchange.process_order(c)
                    if self.pennyjumper.quote_collector:
                        for q in self.pennyjumper.quote_collector:
                            self.exchange.process_order(q)
                    top_of_book = self.exchange.report_top_of_book(current_time)
            if not current_time % write_interval:
                self.exchange.order_history_to_h5(self.h5filename)
                self.exchange.sip_to_h5(self.h5filename)
                
    def qTakeToh5(self):
        temp_df = pd.DataFrame({'qt_take': self.q_take, 'lambda_t': self.lambda_t})
        temp_df.to_hdf(self.h5filename, 'qtl', append=True, format='table', complevel=5, complib='blosc')
        
    def mmProfitabilityToh5(self):
        for m in self.marketmakers:
            temp_df = pd.DataFrame(m.cash_flow_collector)
            temp_df.to_hdf(self.h5filename, 'mmp', append=True, format='table', complevel=5, complib='blosc')
    