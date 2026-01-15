import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from groq import Groq

# Using the NSL-KDD dataset from Kaggle (download KDDTrain+.txt and rename it)
filename = r"C:\Users\nikhi\OneDrive\Desktop\AICTE Cybersecurity Major Project\data\KDDTrain+.txt"

st.set_page_config(page_title="NetSec Analyzer", layout="wide")

st.title("🛡️ NetSec : Network Intrusion Detection System")
st.markdown("### Final Major Project: ML Model Comparison on NSL-KDD Data")

@st.cache_data
def get_data(path):
    cols = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 
        'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 
        'num_failed_logins', 'logged_in', 'num_compromised', 
        'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
        'num_shells', 'num_access_files', 'num_outbound_cmds', 
        'is_host_login', 'is_guest_login', 'count', 'srv_count', 
        'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
        'dst_host_srv_rerror_rate', 'class', 'difficulty'
    ]
    
    try:
        df = pd.read_csv(path, names=cols, nrows=25000)
        df.drop('difficulty', axis=1, inplace=True)

        encoder = LabelEncoder()
        to_encode = ['protocol_type', 'service', 'flag', 'class']
        for c in to_encode:
            df[c] = encoder.fit_transform(df[c])
            
        return df, encoder
    except:
        return None, None

def train_system(df):
    y = df['class']
    X = df.drop('class', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

    model1 = DecisionTreeClassifier(max_depth=10)
    model1.fit(X_train, y_train)
    p1 = model1.predict(X_test)
    acc1 = accuracy_score(y_test, p1)

    model2 = RandomForestClassifier(n_estimators=20, max_depth=10)
    model2.fit(X_train, y_train)
    p2 = model2.predict(X_test)
    acc2 = accuracy_score(y_test, p2)

    final_model = model2 if acc2 > acc1 else model1

    return {
        "dt_acc": acc1,
        "rf_acc": acc2,
        "model": final_model,
        "test_x": X_test,
        "test_y": y_test
    }

st.sidebar.header("Controls")
api_input = st.sidebar.text_input("Groq API Key", type="password")

df, enc = get_data(filename)

if df is None:
    st.error(f"File not found: {filename}. Please download KDDTrain+.txt from Kaggle and rename it.")
else:
    st.sidebar.success(f"Loaded {len(df)} rows")

    if st.sidebar.button("Start Training"):
        with st.spinner("Processing..."):
            res = train_system(df)
            st.session_state['data'] = res
            st.sidebar.success("Done!")

t1, t2, t3 = st.tabs(["📊 Results", "🔎 Live Test", "🤖 AI Chat"])

with t1:
    if 'data' in st.session_state:
        d = st.session_state['data']
        
        c1, c2 = st.columns(2)
        c1.metric("Decision Tree", f"{d['dt_acc']:.2%}")
        c2.metric("Random Forest", f"{d['rf_acc']:.2%}")
        
        graph_df = pd.DataFrame({
            "Algorithm": ["Decision Tree", "Random Forest"],
            "Score": [d['dt_acc'], d['rf_acc']]
        })
        st.plotly_chart(px.bar(graph_df, x="Algorithm", y="Score", color="Algorithm"), use_container_width=True)
        
        better = "Random Forest" if d['rf_acc'] > d['dt_acc'] else "Decision Tree"
        st.info(f"Using {better} for predictions.")
    else:
        st.info("Please click Start Training in the sidebar.")

with t2:
    if 'data' in st.session_state:
        st.subheader("Packet Simulator")
        
        if st.button("Pick Random Packet"):
            tx = st.session_state['data']['test_x']
            ty = st.session_state['data']['test_y']
            
            i = np.random.randint(0, len(tx))
            row = tx.iloc[i]
            label = ty.iloc[i]
            
            m = st.session_state['data']['model']
            out = m.predict([row])[0]
            
            st.session_state['row'] = row
            st.session_state['out'] = out
            st.session_state['lbl'] = label
        
        if 'row' in st.session_state:
            c_a, c_b = st.columns(2)
            
            with c_a:
                st.write("Feature Data:")
                st.dataframe(st.session_state['row'], use_container_width=True)
            
            with c_b:
                st.write("Model Result:")
                val = st.session_state['out']
                real = st.session_state['lbl']
                
                if val == real:
                    st.success(f"Matched: Class {val}")
                else:
                    st.warning(f"Mismatch: Predicted {val}, Actual {real}")

with t3:
    st.subheader("Groq Explanation")
    
    if 'row' in st.session_state:
        if not api_input:
            st.warning("Need API Key.")
        else:
            if st.button("Ask AI"):
                try:
                    client = Groq(api_key=api_input)
                    
                    details = st.session_state['row'].to_string()
                    p_val = st.session_state['out']
                    
                    msg = f"""
                    You are a security analyst.
                    Explain this NSL-KDD packet to a student.
                    Predicted Class: {p_val}
                    
                    Data:
                    {details}
                    
                    Explain:
                    1. Is it an attack?
                    2. What looks suspicious?
                    3. Keep it simple.
                    """
                    
                    with st.spinner("Thinking..."):
                        chat = client.chat.completions.create(
                            messages=[{"role": "user", "content": msg}],
                            model="llama-3.3-70b-versatile"
                        )
                        st.markdown(chat.choices[0].message.content)
                except Exception as e:
                    st.error(str(e))