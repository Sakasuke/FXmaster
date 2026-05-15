//+------------------------------------------------------------------+
//|                                       MultiTF_EMA_RSI_EA.mq5     |
//|                        Multi-Timeframe EMA + RSI Trend EA         |
//|                                                                    |
//|  Strategy:                                                         |
//|    H4 EMA(75) for trend direction filter                          |
//|    M15 RSI(14) for entry trigger                                  |
//|    Trailing stop for trade management                             |
//+------------------------------------------------------------------+
#property copyright "FXmaster"
#property link      ""
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//+------------------------------------------------------------------+
//| Input Parameters                                                  |
//+------------------------------------------------------------------+
input long    MagicNumber       = 654321;
input double  Lots              = 0.1;
input double  StopLoss_Pips     = 30.0;
input double  TakeProfit_Pips   = 50.0;
input int     RSI_Period        = 14;
input double  RSI_Oversold      = 30.0;
input double  RSI_Overbought    = 70.0;
input int     EMA_Period        = 75;
input bool    UseTrailingStop   = true;
input double  TrailingStart_Pips = 20.0;
input double  TrailingStep_Pips  = 5.0;

//+------------------------------------------------------------------+
//| Global Variables                                                  |
//+------------------------------------------------------------------+
CTrade         trade;
int            handleEMA_H4;
int            handleRSI_M15;
datetime       lastBarTime_M15 = 0;
double         pipValue;       // price distance of 1 pip
int            pipDigits;      // decimal digits that represent 1 pip

//+------------------------------------------------------------------+
//| Expert initialization                                             |
//+------------------------------------------------------------------+
int OnInit()
{
   // --- Pip calculation for 4-digit vs 5-digit brokers ---
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   if(digits == 3 || digits == 5)
   {
      pipValue  = _Point * 10.0;
      pipDigits = 1;
   }
   else if(digits == 2 || digits == 4)
   {
      pipValue  = _Point;
      pipDigits = 0;
   }
   else
   {
      // Metals / crypto with unusual digits — fallback
      pipValue  = _Point * 10.0;
      pipDigits = 1;
   }

   // --- Create indicator handles ---
   handleEMA_H4 = iMA(_Symbol, PERIOD_H4, EMA_Period, 0, MODE_EMA, PRICE_CLOSE);
   if(handleEMA_H4 == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create H4 EMA handle");
      return INIT_FAILED;
   }

   handleRSI_M15 = iRSI(_Symbol, PERIOD_M15, RSI_Period, PRICE_CLOSE);
   if(handleRSI_M15 == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create M15 RSI handle");
      return INIT_FAILED;
   }

   // --- Configure CTrade ---
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(10);
   trade.SetTypeFilling(ORDER_FILLING_FOK);

   Print("MultiTF_EMA_RSI_EA initialized successfully");
   Print("Pip value = ", DoubleToString(pipValue, _Digits),
         "  Broker digits = ", IntegerToString(digits));

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(handleEMA_H4 != INVALID_HANDLE)
      IndicatorRelease(handleEMA_H4);
   if(handleRSI_M15 != INVALID_HANDLE)
      IndicatorRelease(handleRSI_M15);

   Print("MultiTF_EMA_RSI_EA removed. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   // --- Trailing stop runs every tick ---
   if(UseTrailingStop)
      ManageTrailingStop();

   // --- Entry logic only on new M15 bar ---
   if(!IsNewBar(PERIOD_M15))
      return;

   // --- Skip if we already have a position with this magic ---
   if(HasOpenPosition())
      return;

   // --- Read H4 EMA ---
   double emaH4[];
   ArraySetAsSeries(emaH4, true);
   if(CopyBuffer(handleEMA_H4, 0, 0, 2, emaH4) < 2)
   {
      Print("WARNING: CopyBuffer failed for H4 EMA");
      return;
   }

   // --- Read H4 close price for trend comparison ---
   MqlRates ratesH4[];
   ArraySetAsSeries(ratesH4, true);
   if(CopyRates(_Symbol, PERIOD_H4, 0, 2, ratesH4) < 2)
   {
      Print("WARNING: CopyRates failed for H4");
      return;
   }

   double closeH4    = ratesH4[0].close;
   double emaValueH4 = emaH4[0];

   // --- Determine trend direction ---
   int trendDirection = 0;  // 0 = neutral
   if(closeH4 > emaValueH4)
      trendDirection = +1;  // bullish
   else if(closeH4 < emaValueH4)
      trendDirection = -1;  // bearish

   if(trendDirection == 0)
      return;

   // --- Read M15 RSI (bar [1] = last closed bar, bar [2] = bar before) ---
   double rsiM15[];
   ArraySetAsSeries(rsiM15, true);
   if(CopyBuffer(handleRSI_M15, 0, 0, 3, rsiM15) < 3)
   {
      Print("WARNING: CopyBuffer failed for M15 RSI");
      return;
   }

   double rsiCurrent  = rsiM15[1];  // last closed bar
   double rsiPrevious = rsiM15[2];  // bar before that

   // --- Entry conditions ---
   bool buySignal  = false;
   bool sellSignal = false;

   // BUY: H4 bullish + RSI was/is below oversold level
   if(trendDirection == +1)
   {
      // RSI crossed below oversold and turned back up, OR RSI is below oversold on bar close
      if(rsiCurrent < RSI_Oversold || (rsiPrevious < RSI_Oversold && rsiCurrent > rsiPrevious))
         buySignal = true;
   }

   // SELL: H4 bearish + RSI was/is above overbought level
   if(trendDirection == -1)
   {
      // RSI crossed above overbought and turned back down, OR RSI is above overbought on bar close
      if(rsiCurrent > RSI_Overbought || (rsiPrevious > RSI_Overbought && rsiCurrent < rsiPrevious))
         sellSignal = true;
   }

   // --- Execute trades ---
   if(buySignal)
      ExecuteBuy();
   else if(sellSignal)
      ExecuteSell();
}

//+------------------------------------------------------------------+
//| Check if a new bar has opened on the specified timeframe          |
//+------------------------------------------------------------------+
bool IsNewBar(ENUM_TIMEFRAMES timeframe)
{
   datetime barTime[];
   if(CopyTime(_Symbol, timeframe, 0, 1, barTime) < 1)
      return false;

   if(barTime[0] != lastBarTime_M15)
   {
      lastBarTime_M15 = barTime[0];
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Check if there is already an open position with our Magic Number |
//+------------------------------------------------------------------+
bool HasOpenPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(PositionGetInteger(POSITION_MAGIC) == MagicNumber &&
         PositionGetString(POSITION_SYMBOL) == _Symbol)
         return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Execute a BUY market order                                       |
//+------------------------------------------------------------------+
void ExecuteBuy()
{
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   if(ask == 0.0)
      return;

   double sl = NormalizeDouble(ask - StopLoss_Pips * pipValue, _Digits);
   double tp = NormalizeDouble(ask + TakeProfit_Pips * pipValue, _Digits);

   if(!trade.Buy(Lots, _Symbol, ask, sl, tp, "MTF_EMA_RSI BUY"))
   {
      Print("ERROR: Buy order failed. Code=", trade.ResultRetcode(),
            " Desc=", trade.ResultRetcodeDescription());
   }
   else
   {
      Print("BUY executed @ ", DoubleToString(ask, _Digits),
            " SL=", DoubleToString(sl, _Digits),
            " TP=", DoubleToString(tp, _Digits));
   }
}

//+------------------------------------------------------------------+
//| Execute a SELL market order                                      |
//+------------------------------------------------------------------+
void ExecuteSell()
{
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if(bid == 0.0)
      return;

   double sl = NormalizeDouble(bid + StopLoss_Pips * pipValue, _Digits);
   double tp = NormalizeDouble(bid - TakeProfit_Pips * pipValue, _Digits);

   if(!trade.Sell(Lots, _Symbol, bid, sl, tp, "MTF_EMA_RSI SELL"))
   {
      Print("ERROR: Sell order failed. Code=", trade.ResultRetcode(),
            " Desc=", trade.ResultRetcodeDescription());
   }
   else
   {
      Print("SELL executed @ ", DoubleToString(bid, _Digits),
            " SL=", DoubleToString(sl, _Digits),
            " TP=", DoubleToString(tp, _Digits));
   }
}

//+------------------------------------------------------------------+
//| Trailing Stop Management                                         |
//|                                                                    |
//| For each open position with our Magic:                            |
//|   - Once profit >= TrailingStart_Pips, begin trailing             |
//|   - Move SL by TrailingStep_Pips only when price moves favorably |
//+------------------------------------------------------------------+
void ManageTrailingStop()
{
   double trailStartPrice = TrailingStart_Pips * pipValue;
   double trailStepPrice  = TrailingStep_Pips  * pipValue;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber)
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;

      long posType     = PositionGetInteger(POSITION_TYPE);
      double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
      double currentSL = PositionGetDouble(POSITION_SL);
      double currentTP = PositionGetDouble(POSITION_TP);

      if(posType == POSITION_TYPE_BUY)
      {
         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         double profit = bid - openPrice;

         // Only start trailing after minimum profit threshold
         if(profit < trailStartPrice)
            continue;

         // New SL is current price minus trailing step
         double newSL = NormalizeDouble(bid - trailStepPrice, _Digits);

         // Only move SL up (never down) and only if it improves by at least 1 point
         if(newSL > currentSL + _Point)
         {
            if(!trade.PositionModify(ticket, newSL, currentTP))
            {
               Print("WARNING: Trailing SL modify failed for BUY #", ticket,
                     " Code=", trade.ResultRetcode());
            }
         }
      }
      else if(posType == POSITION_TYPE_SELL)
      {
         double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         double profit = openPrice - ask;

         if(profit < trailStartPrice)
            continue;

         double newSL = NormalizeDouble(ask + trailStepPrice, _Digits);

         // Only move SL down (never up) for SELL positions
         if(currentSL == 0.0 || newSL < currentSL - _Point)
         {
            if(!trade.PositionModify(ticket, newSL, currentTP))
            {
               Print("WARNING: Trailing SL modify failed for SELL #", ticket,
                     " Code=", trade.ResultRetcode());
            }
         }
      }
   }
}
//+------------------------------------------------------------------+
