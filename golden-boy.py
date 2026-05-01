import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import asyncio
import nest_asyncio
from datetime import datetime
import warnings
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import io

# إعدادات البيئة والجودة
nest_asyncio.apply()
nltk.download('vader_lexicon', quiet=True)
warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# ⚙️ الإعدادات المتقدمة
TOKEN = "8241836194:AAFMothXv8vqpOmF8KRSZB5yp4FUfzFH4gg"
NEWS_API_KEY = "d5e3e2bd811744e694852f58411d82c0"
GOLD_SYMBOL = "GC=F"
DXY_SYMBOL = "DX-Y.NYB"

# ═══════════════════════════════════════════════════════════════
# المحرك العملاق للذكاء الاصطناعي (The Titan Engine)
# ═══════════════════════════════════════════════════════════════

class GoldenBoyTitan:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.eco_model = RandomForestRegressor(n_estimators=20)
        self._initialize_ai()

    def _initialize_ai(self):
        """تدريب أولي للنموذج الاقتصادي على سيناريوهات معقدة"""
        X = np.array([[5.5, 3.1, 104.0], [2.0, 5.0, 100.0], [1.0, 2.0, 95.0], [5.0, 2.0, 106.0]], dtype=float)
        y = np.array([0.2, 0.8, 0.9, 0.1], dtype=float)
        self.eco_model.fit(X, y)

    def force_float(self, val):
        """تحويل أي مدخل إلى رقم عشري نقي لمنع أخطاء المصفوفات"""
        try:
            if isinstance(val, (pd.Series, np.ndarray)):
                return float(val.iloc[-1] if hasattr(val, 'iloc') else val[-1])
            return float(val)
        except:
            return 0.0

    def get_correlations(self):
        """تحليل العلاقة مع مؤشر الدولار الأمريكي"""
        try:
            dxy = yf.download(DXY_SYMBOL, period="1d", interval="5m", progress=False)
            if isinstance(dxy.columns, pd.MultiIndex): dxy.columns = [col[0] for col in dxy.columns]
            dxy_price = self.force_float(dxy['Close'])
            dxy_start = self.force_float(dxy['Close'].iloc[0])
            dxy_change = ((dxy_price - dxy_start) / dxy_start) * 100
            return dxy_price, dxy_change
        except:
            return 104.0, 0.0

    def calculate_indicators(self, df):
        """إضافة مؤشرات احترافية: Bollinger Bands & ATR"""
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['STD'] = df['Close'].rolling(window=20).std()
        df['Upper'] = df['MA20'] + (df['STD'] * 2)
        df['Lower'] = df['MA20'] - (df['STD'] * 2)
        
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        
        tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        return df.dropna()

    def get_market_news(self):
        """جلب وتحليل الأخبار الحقيقية من News API"""
        try:
            url = f"https://newsapi.org/v2/everything?q=gold+market+OR+fed+rate&language=en&sortBy=publishedAt&pageSize=1&apiKey={NEWS_API_KEY}"
            data = requests.get(url, timeout=5).json()
            if data.get('articles'):
                title = data['articles'][0]['title']
                score = self.sia.polarity_scores(title)['compound']
                return title, score
            return "Stable market conditions reported.", 0.0
        except:
            return "News feed unavailable.", 0.0

    def detect_liquidity_zones(self, df):
        """تحديد مناطق السيولة وانفجار السعر"""
        recent = df.tail(50)
        resistance = self.force_float(recent['High'].max())
        support = self.force_float(recent['Low'].min())
        curr = self.force_float(df['Close'])
        
        vol_squeeze = "ضيق (انفجار قريب) 🧨" if (self.force_float(recent['Upper']) - self.force_float(recent['Lower'])) < 2 else "طبيعي 🌊"
        
        status = "هادئ"
        if curr >= (resistance * 0.999): status = "منطقة بيع مؤسسي ⚠️"
        elif curr <= (support * 1.001): status = "منطقة تجميع حيتان 🐋"
        
        return resistance, support, status, vol_squeeze

    def generate_chart_img(self):
        """نظام الصورة: رسم مخطط بياني مع إشارات البيع والشراء"""
        df = yf.download(GOLD_SYMBOL, period="1d", interval="15m", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
        
        # جلب التحليل لوضع الإشارة
        analysis = self.full_scan()
        
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df['Close'], label='Price', color='#FFD700', linewidth=2)
        
        # إضافة إشارات شراء/بيع على آخر نقطة
        last_time = df.index[-1]
        last_price = df['Close'].iloc[-1]
        
        if analysis['prob'] > 0.60:
            plt.scatter(last_time, last_price, color='lime', marker='^', s=200, label='Titan BUY Signal')
            plt.annotate("BUY", (last_time, last_price), textcoords="offset points", xytext=(0,10), ha='center', color='lime', fontweight='bold')
        elif analysis['prob'] < 0.40:
            plt.scatter(last_time, last_price, color='red', marker='v', s=200, label='Titan SELL Signal')
            plt.annotate("SELL", (last_time, last_price), textcoords="offset points", xytext=(0,-20), ha='center', color='red', fontweight='bold')

        plt.title(f"XAUUSD LIVE CHART - {datetime.now().strftime('%H:%M')}")
        plt.legend()
        plt.grid(True, alpha=0.2)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf

    def full_scan(self):
        """التحليل الشامل النهائي مع إصلاح أخطاء الـ Shape"""
        df = yf.download(GOLD_SYMBOL, period="5d", interval="5m", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
        df = self.calculate_indicators(df)
        
        # الذكاء الاصطناعي الفني
        X_tech = df[['Close', 'RSI', 'EMA_20', 'Volume']].tail(100)
        y_tech = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)[-100:]
        model = xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.05, verbosity=0)
        model.fit(X_tech.values[:-1], y_tech[:-1])
        prob_tech = float(model.predict_proba(X_tech.values[-1:]) [0][1])

        # الربط الاقتصادي والدولار
        dxy_p, dxy_c = self.get_correlations()
        news, n_score = self.get_market_news()
        
        # إصلاح خطأ المصفوفة هنا
        eco_input = np.array([[5.5, 3.2, float(dxy_p)]], dtype=float)
        prob_eco = float(self.eco_model.predict(eco_input)[0])
        
        final_prob = (prob_tech * 0.4) + ((n_score+1)/2 * 0.3) + (prob_eco * 0.3)
        
        atr = self.force_float(df['ATR'])
        curr = self.force_float(df['Close'])
        res, sup, l_stat, squeeze = self.detect_liquidity_zones(df)
        
        return {
            'price': curr, 'prob': final_prob, 'tech': prob_tech,
            'news': news, 'dxy': dxy_p, 'dxy_ch': dxy_c,
            'res': res, 'sup': sup, 'l_stat': l_stat, 'squeeze': squeeze,
            'sl_buy': curr - (atr*2.5), 'tp_buy': curr + (atr*2),
            'sl_sell': curr + (atr*2.5), 'tp_sell': curr - (atr*2)
        }

# ═══════════════════════════════════════════════════════════════
# واجهة التحكم ونظام التنبيه التلقائي
# ═══════════════════════════════════════════════════════════════

def get_pro_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔥 تحليل التيتان الشامل", callback_data='scan_pro')],
        [InlineKeyboardButton("📸 عرض صورة الشارت", callback_data='send_chart')],
        [InlineKeyboardButton("💵 مؤشر الدولار DXY", callback_data='dxy'), InlineKeyboardButton("📊 السيولة", callback_data='liq')],
        [InlineKeyboardButton("⚙️ إدارة اللوت والمخاطر", callback_data='risk_tool')]
    ])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🛡 **GOLDEN BOY TITAN V22.0 (STABLE)**\n\n"
        "تم إصلاح كافة أخطاء المصفوفات وتحسين سرعة الاستجابة.\n"
        "المحركات: XGBoost, Random Forest, Sentiment, DXY-Corr.",
        reply_markup=get_pro_keyboard(), parse_mode='Markdown'
    )

async def handle_interaction(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    titan = GoldenBoyTitan()

    if query.data == 'scan_pro':
        m = await query.message.reply_text("🌀 جاري استدعاء البيانات الكلية وتحليل الأنماط...")
        try:
            res = titan.full_scan()
            signal = "BUY 🟢" if res['prob'] > 0.60 else "SELL 🔴" if res['prob'] < 0.40 else "WAIT ⚖️"
            
            report = (
                f"🎯 **إشارة التيتان:** `{signal}`\n"
                f"━━━━━━━━━━━━━━━\n"
                f"💰 **السعر:** `${res['price']:.2f}`\n"
                f"📉 **الدولار (DXY):** `{res['dxy']:.2f}` ({res['dxy_ch']:.2f}%)\n\n"
                f"🛡 **إدارة المخاطر:**\n"
                f"├ **SL:** `{res['sl_buy' if 'BUY' in signal else 'sl_sell']:.2f}`\n"
                f"└ **TP:** `{res['tp_buy' if 'BUY' in signal else 'tp_sell']:.2f}`\n\n"
                f"🔍 **رادار السيولة:**\n"
                f"├ السقف: `{res['res']:.1f}` | القاع: `{res['sup']:.1f}`\n"
                f"└ الوضع: `{res['l_stat']}` | `{res['squeeze']}`\n\n"
                f"📰 **خبر الساعة:**\n_{res['news']}_\n"
                f"━━━━━━━━━━━━━━━\n"
                f"🔥 **ثقة النظام:** `{res['prob']*100:.1f}%`"
            )
            await m.edit_text(report, parse_mode='Markdown', reply_markup=get_pro_keyboard())
        except Exception as e:
            await m.edit_text(f"❌ خطأ داخلي: {str(e)}")

    elif query.data == 'send_chart':
        m = await query.message.reply_text("🎨 جاري رسم المخطط...")
        img = titan.generate_chart_img()
        await query.message.reply_photo(photo=img, caption="📊 شارت الذهب اللحظي مع إشارات التداول.")
        await m.delete()

    elif query.data == 'risk_tool':
        await query.message.reply_text("⚖️ **نصيحة المخاطر:** استخدم لوت 0.01 لكل 100$ من رأس مالك لضمان الأمان.")

    elif query.data == 'dxy':
        p, c = titan.get_correlations()
        await query.message.reply_text(f"💵 **مؤشر الدولار الأمريكي:**\nالسعر الحالي: `{p:.2f}`\nالتغيير اليومي: `{c:.2f}%`", parse_mode='Markdown')

    elif query.data == 'liq':
        df = yf.download(GOLD_SYMBOL, period="1d", interval="5m", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
        df = titan.calculate_indicators(df)
        r, s, st, sq = titan.detect_liquidity_zones(df)
        await query.message.reply_text(f"🔍 **تقرير السيولة:**\nالمقاومة القصوى: `{r:.2f}`\nالدعم الأقصى: `{s:.2f}`\nحالة الحيتان: `{st}`", parse_mode='Markdown')

async def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(handle_interaction))
    
    print("💎 Titan System is Online and Stable...")
    
    await app.initialize()
    await app.bot.delete_webhook(drop_pending_updates=True)
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)
    
    while True:
        await asyncio.sleep(1)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except:
        pass
