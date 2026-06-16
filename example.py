import win32evtlog, win32con, win32api, queue, threading, xml.etree.ElementTree as ET

event_queue = queue.Queue()

def log_watcher():
    QUERY = "*[System[(EventID=4625 or EventID=4624 or EventID=4648)]]"
    handle = win32evtlog.EvtSubscribe(
        "Security", win32evtlog.EvtSubscribeToFutureEvents,
        Query=QUERY, SignalEvent=None
    )
    while True:
        events = win32evtlog.EvtNext(handle, 10, 1000)
        for e in events:
            xml_str = win32evtlog.EvtRender(e, win32evtlog.EvtRenderEventXml)
            event_queue.put(parse_event(xml_str))

def parse_event(xml_str):
    root = ET.fromstring(xml_str)
    ns = {'e': 'http://schemas.microsoft.com/win/2004/08/events/event'}
    eid = root.find('.//e:EventID', ns).text
    ts  = root.find('.//e:TimeCreated', ns).attrib['SystemTime']
    data = {d.attrib.get('Name'): d.text for d in root.findall('.//e:Data', ns)}
    return {
        'event_id': eid, 'timestamp': ts,
        'username': data.get('TargetUserName'),
        'src_ip':   data.get('IpAddress'),
        'status':   'fail' if eid == '4625' else 'success'
    }

threading.Thread(target=log_watcher, daemon=True).start()



from collections import defaultdict, deque
import numpy as np, joblib, time

model_rf = joblib.load('rf_model.pkl')          # supervised
model_if = joblib.load('isolation_forest.pkl')  # unsupervised

# Buffer: 60 giây gần nhất per user
user_buffer = defaultdict(lambda: deque(maxlen=500))

def extract_features(username, buffer, window_sec=60):
    now = time.time()
    recent = [e for e in buffer if now - e['ts'] <= window_sec]
    if len(recent) < 2:
        return None
    fails = sum(1 for e in recent if e['status'] == 'fail')
    return {
        'login_freq':    len(recent),
        'fail_ratio':    fails / len(recent),
        'unique_ips':    len(set(e['src_ip'] for e in recent)),
        'off_hours':     int(any(e['hour'] not in range(8,18) for e in recent)),
    }

def process_event(event):
    user = event['username']
    user_buffer[user].append({**event, 'ts': time.time()})
    feats = extract_features(user, user_buffer[user])
    if feats is None:
        return None
    X = np.array([[feats['login_freq'], feats['fail_ratio'],
                   feats['unique_ips'], feats['off_hours']]])
    score_rf = model_rf.predict_proba(X)[0][1]       # P(anomaly)
    score_if = -model_if.score_samples(X)[0]         # higher = more anomalous
    return {'user': user, 'score_rf': score_rf, 'score_if': score_if, **feats}


import streamlit as st, pandas as pd, time

st.set_page_config(page_title="Auth Anomaly Monitor", layout="wide")
st.title("🔐 Auth Log Anomaly Detection")

THRESHOLD = 0.7
results_store = []  # shared list, append từ background thread

col1, col2, col3 = st.columns(3)
chart_placeholder  = st.empty()
table_placeholder  = st.empty()

while True:
    df = pd.DataFrame(results_store[-200:])  # 200 events gần nhất
    if not df.empty:
        col1.metric("Events (60s)",  len(df))
        col2.metric("Alerts",        int((df['score_rf'] > THRESHOLD).sum()))
        col3.metric("Avg fail ratio", f"{df['fail_ratio'].mean():.1%}")
        chart_placeholder.line_chart(df.set_index('timestamp')[['score_rf','score_if']])
        alerts = df[df['score_rf'] > THRESHOLD][['user','score_rf','fail_ratio','unique_ips']]
        table_placeholder.dataframe(alerts.style.highlight_max(color='#ffcccc'), use_container_width=True)
    time.sleep(5)
    st.rerun()