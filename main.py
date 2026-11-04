import re
import pandas as pd
import streamlit as st  # ⭕ 반드시 st 데코레이터 사용 전에 임포트되어야 합니다.
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# 페이지 기본 설정
st.set_page_config(page_title="급식 규칙 찾기", layout="wide")
st.title("🍽️ 송탄고 급식 연관 규칙 분석") 
@st.cache_data
def generate_rules(transactions):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # 1. max_len 제한 없이 모든 빈발 항목집합 생성 (단일 항목 포함)
    frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)
    
    if frequent_itemsets.empty:
        return pd.DataFrame()
        
    # 2. 전체 빈발 항목집합을 전달하여 연관 규칙 생성
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.0, support_only=False)
    
    if rules.empty:
        return pd.DataFrame()

    # 3. 규칙 생성 후 조건(1개)과 결과(1개)로 이루어진 '메뉴 쌍' 규칙만 필터링
    rules = rules[
        (rules["antecedents"].apply(lambda x: len(x) == 1)) & 
        (rules["consequents"].apply(lambda x: len(x) == 1))
    ].copy()

    # 조건/결과 텍스트 변환 및 동시 출현 횟수 계산
    rules["조건"] = rules["antecedents"].apply(lambda x: list(x)[0])
    rules["결과"] = rules["consequents"].apply(lambda x: list(x)[0])
    rules["동시"] = (rules["support"] * len(transactions)).round().astype(int)
    
    rules["지지도"] = rules["support"]
    rules["신뢰도"] = rules["confidence"]
    rules["향상도"] = rules["lift"]
    
    result_df = rules[["조건", "결과", "동시", "지지도", "신뢰도", "향상도"]].copy()
    return result_df
