# distutils: language = c++

import random
import time

cimport cython

import numpy as np
cimport numpy as np
import pandas as pd

cimport mmabm.orderbookcpp as orderbook
cimport mmabm.traderc as trader

from mmabm.sharedc cimport Side, OType, TType


cdef class Runner(object):
    cdef int run_steps, prime1, write_interval, current_time, mpi
    cdef bint provider, taker, informed, pj, marketmaker
    cdef double q_provide, alpha_pj
    cdef public str h5filename
    cdef dict liquidity_providers
    
    cdef np.ndarray provider_array, taker_array, marketmakers, q_take, lambda_t, takerTradeV, traders
    
    cdef orderbook.Orderbook exchange
    
    def __init__(self, h5filename='test.h5', mpi=1, prime1=20, run_steps=100000, write_interval=5000, **kwargs):
        self.exchange = orderbook.Orderbook()
        self.h5filename = h5filename
        self.mpi = mpi
        self.run_steps = run_steps + 1
        self.liquidity_providers = {}
        self.provider = kwargs.pop('Provider')
        if self.provider:
            self.provider_array = self.buildProviders(kwargs['numProviders'], kwargs['providerMaxQ'],
                                                      kwargs['pAlpha'], kwargs['pDelta'])
            self.q_provide = kwargs['qProvide']
        self.taker = kwargs.pop('Taker')
        if self.taker:
            self.taker_array = self.buildTakers(kwargs['numTakers'], kwargs['takerMaxQ'], kwargs['tMu'])
        self.informed = kwargs.pop('InformedTrader')
        if self.informed:
            if self.taker:
                takerTradeV = np.array([t[1] for t in self.taker_array])
            informedTrades = np.int(kwargs['iMu']*np.sum(takerTradeV*self.run_steps/self.t_delta_t) if self.taker else 1/kwargs['iMu'])
            self.t_delta_i, self.informed_trader = self.buildInformedTrader(kwargs['informedMaxQ'], kwargs['informedRunLength'], informedTrades, prime1)
        self.pj = kwargs.pop('PennyJumper')
        if self.pj:
            self.pennyjumper = self.buildPennyJumper()
            self.alpha_pj = kwargs['AlphaPJ']
        self.marketmaker = kwargs.pop('MarketMaker')
        if self.marketmaker:
            self.marketmakers = self.buildMarketMakers(kwargs['MMMaxQ'], kwargs['NumMMs'], kwargs['MMQuotes'], 
                                                       kwargs['MMQuoteRange'], kwargs['MMDelta'])
        self.traders = self.makeAll()
        self.q_take, self.lambda_t = self.makeQTake(kwargs['QTake'], kwargs['Lambda0'], kwargs['WhiteNoise'], kwargs['CLambda'])
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
        provider_ids = [1000 + i for i in range(numProviders)]
        provider_list = [trader.Provider(p, providerMaxQ, pDelta) for p in provider_ids]
        self.liquidity_providers.update(dict(zip(provider_ids, provider_list)))
        provider_size = np.array([p.quantity for p in provider_list])
        t_delta_p = np.floor(np.random.exponential(1/pAlpha, numProviders)+1)*provider_size
        return np.vstack([np.array(provider_list), t_delta_p.astype(np.int)]).T
    
    def buildTakers(self, numTakers, takerMaxQ, tMu):
        ''' Takers id starts with 2
        '''
        takers_list = [2000 + i for i in range(numTakers)]
        takers = np.array([trader.Taker(t, takerMaxQ) for t in takers_list])
        taker_size = np.array([t.quantity for t in takers])
        t_delta_t = np.floor(np.random.exponential(1/tMu, numTakers)+1)*taker_size
        return np.vstack([takers, t_delta_t.astype(np.int)]).T
    
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
        return t_delta_i, np.vstack([informed, 0]).T
    
    def buildPennyJumper(self):
        ''' PJ id starts with 4
        '''
        jumper = trader.PennyJumper(4000, 1, self.mpi)
        self.liquidity_providers.update({4000: jumper})
        return jumper

    def buildMarketMakers(self, mMMaxQ, numMMs, mMQuotes, mMQuoteRange, mMDelta):
        ''' MM id starts with 3
        '''
        marketmaker_ids = [3000 + i for i in range(numMMs)]
        marketmaker_list = [trader.MarketMaker(p, mMMaxQ, mMDelta, mMQuotes, mMQuoteRange) for p in marketmaker_ids]
        self.liquidity_providers.update(dict(zip(marketmaker_ids, marketmaker_list)))
        t_delta_m = np.array([m.quantity for m in marketmaker_list])
        return np.vstack([np.array(marketmaker_list), t_delta_m]).T
    
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
        
    def makeAll(self):
        trader_list = []
        if self.provider:
            trader_list.append(self.provider_array)
        if self.taker:
                trader_list.append(self.taker_array)
        if self.marketmaker:
            trader_list.append(self.marketmakers)
        if self.informed:
            trader_list.append(self.informed_trader)
        all_traders = np.vstack(tuple(trader_list))
        return all_traders
    
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
        self.exchange.add_order_to_book(9999, 1, 0, 1, Side.ASK, ba)
        self.exchange.add_order_to_history(qask)
        seed_provider.local_book[2] = qbid
        self.exchange.add_order_to_book(9999, 2, 0, 1, Side.BID, bb)
        self.exchange.add_order_to_history(qbid)
    
    @cython.boundscheck(False)     
    cdef void makeSetup(self, int prime1, float lambda0):
        cdef int current_time
        cdef np.ndarray p
        top_of_book = self.exchange.report_top_of_book(0)
        for current_time in range(1, prime1):
            np.random.shuffle(self.provider_array)
            for p in self.provider_array:
                if not current_time % p[1]:
                    self.exchange.process_order(p[0].process_signalp(current_time, top_of_book, self.q_provide, -lambda0))
                    top_of_book = self.exchange.report_top_of_book(current_time)
    
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
    cdef void runMcs(self, int prime1, int write_interval):
        cdef int current_time
        cdef np.ndarray row
        top_of_book = self.exchange.report_top_of_book(prime1)
        for current_time in range(prime1, self.run_steps):
            np.random.shuffle(self.traders)
            for row in self.traders:
                if row[0].trader_type == TType.Provider:
                    if not current_time % row[1]:
                        self.exchange.process_order(row[0].process_signalp(current_time, top_of_book, self.q_provide, self.lambda_t[current_time]))
                        top_of_book = self.exchange.report_top_of_book(current_time)
                    row[0].bulk_cancel(current_time)
                    if row[0].cancel_collector:
                        self.doCancels(row[0])
                        top_of_book = self.exchange.report_top_of_book(current_time)
                elif row[0].trader_type == TType.MarketMaker:
                    if not current_time % row[1]:
                        row[0].process_signalm(current_time, top_of_book, self.q_provide)
                        for q in row[0].quote_collector:
                            self.exchange.process_order(q)
                        top_of_book = self.exchange.report_top_of_book(current_time)
                    row[0].bulk_cancel(current_time)
                    if row[0].cancel_collector:
                        self.doCancels(row[0])
                        top_of_book = self.exchange.report_top_of_book(current_time)
                elif row[0].trader_type == TType.Taker:
                    if not current_time % row[1]:
                        self.exchange.process_order(row[0].process_signalt(current_time, self.q_take[current_time]))
                        if self.exchange.traded:
                            self.confirmTrades()
                            top_of_book = self.exchange.report_top_of_book(current_time)
                else:
                    if current_time in self.t_delta_i:
                        self.exchange.process_order(row[0].process_signali(current_time))
                        if self.exchange.traded:
                            self.confirmTrades()
                            top_of_book = self.exchange.report_top_of_book(current_time)
            if not current_time % write_interval:
                self.exchange.order_history_to_h5(self.h5filename)
                self.exchange.sip_to_h5(self.h5filename)
                
    @cython.boundscheck(False)            
    cdef void runMcsPJ(self, int prime1, int write_interval):
        cdef int current_time
        cdef np.ndarray row
        top_of_book = self.exchange.report_top_of_book(prime1)
        for current_time in range(prime1, self.run_steps):
            np.random.shuffle(self.traders)
            for row in self.traders:
                if row[0].trader_type == TType.Provider:
                    if not current_time % row[1]:
                        self.exchange.process_order(row[0].process_signalp(current_time, top_of_book, self.q_provide, self.lambda_t[current_time]))
                        top_of_book = self.exchange.report_top_of_book(current_time)
                    row[0].bulk_cancel(current_time)
                    if row[0].cancel_collector:
                        self.doCancels(row[0])
                        top_of_book = self.exchange.report_top_of_book(current_time)
                elif row[0].trader_type == TType.MarketMaker:
                    if not current_time % row[1]:
                        row[0].process_signalm(current_time, top_of_book, self.q_provide)
                        for q in row[0].quote_collector:
                            self.exchange.process_order(q)
                        top_of_book = self.exchange.report_top_of_book(current_time)
                    row[0].bulk_cancel(current_time)
                    if row[0].cancel_collector:
                        self.doCancels(row[0])
                        top_of_book = self.exchange.report_top_of_book(current_time)
                elif row[0].trader_type == TType.Taker:
                    if not current_time % row[1]:
                        self.exchange.process_order(row[0].process_signalt(current_time, self.q_take[current_time]))
                        if self.exchange.traded:
                            self.confirmTrades()
                            top_of_book = self.exchange.report_top_of_book(current_time)
                else:
                    if current_time in self.t_delta_i:
                        self.exchange.process_orderi(row[0].process_signal(current_time))
                        if self.exchange.traded:
                            self.confirmTrades()
                            top_of_book = self.exchange.report_top_of_book(current_time)
                if random.random() < self.alpha_pj:
                    self.pennyjumper.process_signalj(current_time, top_of_book, self.q_take[current_time])
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
            temp_df = pd.DataFrame(m[0].cash_flow_collector)
            temp_df.to_hdf(self.h5filename, 'mmp', append=True, format='table', complevel=5, complib='blosc')
    
    
if __name__ == '__main__':
    
    print(time.time())
    
    settings = {'Provider': True, 'numProviders': 38, 'providerMaxQ': 1, 'pAlpha': 0.0375, 'pDelta': 0.025, 'qProvide': 0.5,
                'Taker': True, 'numTakers': 50, 'takerMaxQ': 1, 'tMu': 0.001,
                'InformedTrader': False, 'informedMaxQ': 1, 'informedRunLength': 1, 'iMu': 0.005,
                'PennyJumper': False, 'AlphaPJ': 0.05,
                'MarketMaker': True, 'NumMMs': 1, 'MMMaxQ': 1, 'MMQuotes': 12, 'MMQuoteRange': 60, 'MMDelta': 0.025,
                'QTake': True, 'WhiteNoise': 0.001, 'CLambda': 10.0, 'Lambda0': 100}
    
    for j in range(51, 61):
        random.seed(j)
        np.random.seed(j)
    
        start = time.time()
        
        h5_root = 'python_pyziabmc_%d' % j
        h5dir = 'C:\\Users\\user\\Documents\\Agent-Based Models\\h5 files\\Trial 901\\'
        h5_file = '%s%s.h5' % (h5dir, h5_root)
    
        market1 = Runner(h5filename=h5_file, **settings)

        print('Run %d: %.1f seconds' % (j, time.time() - start))