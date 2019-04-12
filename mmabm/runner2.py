import random
import time

import numpy as np

import mmabm.learner2 as learner
import mmabm.orderbook as orderbook
import mmabm.trader as trader

from mmabm.settings import *
from mmabm.shared import Side, OType, TType
from mmabm.signal2 import ImbalanceSignal


class Runner:
    
    def __init__(self, h5filename='test.h5', mpi=MPI, prime1=PRIME1, run_steps=RUN_STEPS, write_interval=WRITE_INTERVAL, **kwargs):
        self.exchange = orderbook.Orderbook()
        self.oi_signal = ImbalanceSignal(OI_SIGNAL, OI_HIST_LEN)
        self.h5filename = h5filename
        self.mpi = mpi
        self.run_steps = run_steps + 1
        self.liquidity_providers = {}
        self.traders = []
        if PROVIDER:
            self.num_providers = NUM_PROVIDERS
            self.providers = self.buildProviders(PROVIDER_MAXQ, PROVIDER_ALPHA, PROVIDER_DELTA)
            self.q_provide = Q_PROVIDE
            self.traders.extend(self.providers)
        if TAKER:
            self.takers = self.buildTakers(NUM_TAKERS, TAKER_MAXQ, TAKER_MU)
            self.traders.extend(self.takers)
        if INFORMED:
            informedTrades = np.int(INFORMED_MU*np.sum(np.array([t.quantity*self.run_steps/t.delta_t for t in self.takers])) \
                if TAKER else 1/INFORMED_MU)
            self.informed_trader = self.buildInformedTrader(INFORMED_MAXQ, INFORMED_RUN_LENGTH, informedTrades, prime1)
            self.traders.append(self.informed_trader)
        if PENNYJUMPER:
            self.pennyjumper = self.buildPennyJumper()
            self.alpha_pj = PJ_ALPHA
        if MARKETMAKER:
            self.marketmakers = self.buildMarketMakers(NUM_MMS, MM_MAXQ, ARR_INT, GENETIC_INT)
            self.traders.extend(self.marketmakers)
        self.num_traders = len(self.traders)
        self.q_take, self.lambda_t = self.makeQTake(Q_TAKE, LAMBDA0, WHITENOISE, C_LAMBDA)
        self.seedOrderbook()
        if PROVIDER:
            self.makeSetup(prime1, LAMBDA0)
        else:
            self.prime_MML(1, 1002000, 997995)


    def buildProviders(self, providerMaxQ, pAlpha, pDelta):
        ''' Providers id starts with 1
        '''
        provider_ids = [1000 + i for i in range(self.num_providers)]
        provider_list = [trader.Provider(p, providerMaxQ, pDelta, pAlpha) for p in provider_ids]
        self.liquidity_providers.update(dict(zip(provider_ids, provider_list)))
        return provider_list

    def buildTakers(self, numTakers, takerMaxQ, tMu):
        ''' Takers id starts with 2
        '''
        taker_ids = [2000 + i for i in range(numTakers)]
        return [trader.Taker(t, takerMaxQ, tMu) for t in taker_ids]

    def buildInformedTrader(self, informedMaxQ, informedRunLength, informedTrades, prime1):
        ''' Informed trader id starts with 5
        '''
        return trader.InformedTrader(5000, informedMaxQ, informedTrades, informedRunLength, prime1, self.run_steps)

    def buildPennyJumper(self):
        ''' PJ id starts with 4
        '''
        jumper = trader.PennyJumper(4000, 1, self.mpi)
        self.liquidity_providers.update({4000: jumper})
        return jumper

    def buildMarketMakers(self, numMMs, maxq, arr_int, g_int):
        ''' MM id starts with 3
        '''
        marketmaker_ids = [3000 + i for i in range(numMMs)]
        marketmaker_list = [learner.MarketMakerL(p, maxq, arr_int, g_int) for p in marketmaker_ids]
        self.liquidity_providers.update(dict(zip(marketmaker_ids, marketmaker_list)))
        return marketmaker_list

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

    def seedOrderbook(self):
        seed_provider = trader.Provider(9999, 1, 0.05, 0.025)
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

    def makeSetup(self, prime1, lambda0):
        top_of_book = self.exchange.report_top_of_book(0)
        for current_time in range(1, prime1):
            ps = random.sample(self.providers, self.num_providers)
            for p in ps:
                if not current_time % p.delta_t:
                    self.exchange.process_order(p.process_signal(current_time, top_of_book, self.q_provide, -lambda0))
                    top_of_book = self.exchange.report_top_of_book(current_time)
        ask = top_of_book['best_ask']
        bid = top_of_book['best_bid']
        #self.signal.midl1 = (ask + bid)/2
        self.prime_MML(prime1-1, ask, bid)
        
    def prime_MML(self, step, ask, bid):
        for m in self.marketmakers:
            m.seed_book(step, ask, bid)
            for q in m.quote_collector:
                self.exchange.process_order(q)
        top_of_book = self.exchange.report_top_of_book(step)
        #self.signal.make_mid_signal(step, top_of_book['best_bid'], top_of_book['best_ask'])

    def _make_signals(self, step):
        self.oi_signal.make_signal(step)

    def doCancels(self, trader):
        for c in trader.cancel_collector:
            self.exchange.process_order(c)
                    
    def confirmTrades(self):
        for c in self.exchange.confirm_trade_collector:
            contra_side = self.liquidity_providers[c['trader']]
            contra_side.confirm_trade_local(c)
            #print('Trade', c['timestamp'], ': ', c['price'])
            #self.signal.arrv += c['quantity']
            # side of the resting order: ASK -> taker buy
            self.oi_signal.update_v(c['quantity'] * (1 if c['side'] == Side.ASK else -1))

    def runMcs(self, prime1, write_interval):
        top_of_book = self.exchange.report_top_of_book(prime1)
        for current_time in range(prime1, self.run_steps):
            traders = random.sample(self.traders, self.num_traders)
            for t in traders:
                if t.trader_type == TType.Provider:
                    if not current_time % t.delta_t:
                        self.exchange.process_order(t.process_signal(current_time, top_of_book, self.q_provide, self.lambda_t[current_time]))
                        top_of_book = self.exchange.report_top_of_book(current_time)
                    t.bulk_cancel(current_time)
                    if t.cancel_collector:
                        self.doCancels(t)
                        top_of_book = self.exchange.report_top_of_book(current_time)
                elif t.trader_type == TType.MarketMaker:
                    if not current_time % t.arrInt:
                        self._make_signals(current_time)
                        t.process_signal1(current_time, (self._oi_signal.v, self._oi_signal.str, 
                                                         top_of_book['best_bid'], top_of_book['best_ask']))
                        if t.cancel_collector: # need to check?
                            self.doCancels(t)
                        top_of_book = self.exchange.report_top_of_book(current_time)
                        t.process_signal2(current_time, top_of_book['best_bid'], top_of_book['best_ask']) # <-- Here, 20190410
                        for q in t.quote_collector:
                            self.exchange.process_order(q)
                        if t.cancel_collector: # need to check?
                            self.doCancels(t)
                        top_of_book = self.exchange.report_top_of_book(current_time)
                        self.signal.reset_current() # < -- a general self._reset_signals here?
                elif t.trader_type == TType.Taker:
                    if not current_time % t.delta_t:
                        self.exchange.process_order(t.process_signal(current_time, self.q_take[current_time]))
                        if self.exchange.traded:
                            self.confirmTrades()
                            top_of_book = self.exchange.report_top_of_book(current_time)
                else:
                    if current_time in t.delta_t:
                        self.exchange.process_order(t.process_signal(current_time))
                        if self.exchange.traded:
                            self.confirmTrades()
                            top_of_book = self.exchange.report_top_of_book(current_time)
            if not current_time % write_interval:
                self.exchange.order_history_to_h5(self.h5filename)
                self.exchange.sip_to_h5(self.h5filename)




if __name__ == '__main__':
    
    print(time.time())
    
    for j in range(51, 52):
        random.seed(j)
        np.random.seed(j)
    
        start = time.time()
        
        h5_root = 'python_mmabm_%d_test' % j
        h5dir = 'C:\\Users\\user\\Documents\\Agent-Based Models\\h5 files\\mmabmTests\\'
        h5_file = '%s%s.h5' % (h5dir, h5_root)
    
        market1 = Runner(h5filename=h5_file)
        print(market1.exchange.report_top_of_book(20))

        print('Run %d: %.1f seconds' % (j, time.time() - start))


