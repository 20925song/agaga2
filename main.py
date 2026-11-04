import re
import pandas as pd
import streamlit as st
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# 페이지 기본 설정
st.set_page_config(page_title="급식 규칙 찾기", layout="wide")
st.title("🍽️ 송탄고 급식 연관 규칙 분석")

# 데이터 로드 및 전처리
DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/songtan_meals_184.csv"

@st.cache_data
def load_and_preprocess_data(url):
    df = pd.read_csv(url, encoding="utf-8")
    
    transactions = []
    all_menus = set()
    
    for raw_menu in df["메뉴"].dropna():
        # 메뉴 이름에서 괄호 및 괄호 안 내용 제거, 양끝 공백 제거
        items = [re.sub(r"\(.*?\)", "", item).strip() for item in str(raw_menu).split("|")]
        # 빈 문자열 제외 및 중복 제거
        items = list(set([item for item in items if item]))
        
        if items:
            transactions.append(items)
            all_menus.update(items)
            
    return df, transactions, sorted(list(all_menus))

@st.cache_data
def generate_rules(transactions):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # 최소 지지도 설정 (184일 중 최소 2번 이상 동시 출현: 2/184 ≈ 0.01)
    frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True, max_len=2)
    
    # 지지도가 2개 이상인 항목 조합만 필터링 (메뉴 쌍)
    frequent_itemsets = frequent_itemsets[frequent_itemsets["itemsets"].apply(lambda x: len(x) == 2)]
    
    if frequent_itemsets.empty:
        return pd.DataFrame()
        
    # 연관 규칙 생성
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.0, support_only=False)
    
    # mlxtend 0.23+ 버전과의 호환성을 위해 향상도 계산 보장
    if "lift" not in rules.columns:
        rules["lift"] = rules["confidence"] / rules["consequent support"]

    # 조건/결과 텍스트 변환 및 동시 출현 횟수 계산
    rules["조건"] = rules["antecedents"].apply(lambda x: list(x)[0])
    rules["결과"] = rules["consequents"].apply(lambda x: list(x)[0])
    rules["동시"] = (rules["support"] * len(transactions)).round().astype(int)
    
    rules["지지도"] = rules["support"]
    rules["신뢰도"] = rules["confidence"]
    rules["향상도"] = rules["lift"]
    
    result_df = rules[["조건", "결과", "동시", "지지도", "신뢰도", "향상도"]].copy()
    return result_df

# 데이터 로딩
raw_df, transactions, menu_list = load_and_preprocess_data(DATA_URL)
rules_df = generate_rules(transactions)

# 기본 통계 계산
total_days = len(transactions)
total_menu_types = len(menu_list)

# 중복 없는 무방향 메뉴 쌍 수 계산
if not rules_df.empty:
    unique_pairs_count = len(rules_df[['조건', '결과']].apply(lambda x: tuple(sorted(x)), axis=1).unique())
else:
    unique_pairs_count = 0

# 상단 요약 정보 출력
st.markdown(f"**급식 일수:** {total_days}일 | **메뉴 종류 수:** {total_menu_types}개 | **함께 나온 적이 있는 메뉴 쌍의 수:** {unique_pairs_count}쌍")
st.divider()

# 컨트롤 러아웃 (사이드바 또는 상단)
col1, col2 = st.columns([1, 1])

with col1:
    selected_menu = st.selectbox("특정 메뉴가 포함된 규칙만 보기", ["전체 보기"] + menu_list)

with col2:
    sort_option = st.selectbox("정렬 기준 선택", ["향상도 순", "신뢰도 순", "동시 순"])

# 데이터 필터링
filtered_df = rules_df.copy()
if selected_menu != "전체 보기":
    filtered_df = filtered_df[(filtered_df["조건"] == selected_menu) | (filtered_df["결과"] == selected_menu)]

# 데이터 정렬
sort_mapping = {
    "향상도 순": "향상도",
    "신뢰도 순": "신뢰도",
    "동시 순": "동시"
}
filtered_df = filtered_df.sort_values(by=sort_mapping[sort_option], ascending=False).reset_index(drop=True)

# 시각화 (향상도 상위 10개 가로 막대그래프)
st.subheader("📊 향상도 상위 10개 규칙")
if not filtered_df.empty:
    top_10_df = filtered_df.sort_values(by="향상도", ascending=False).head(10).copy()
    top_10_df["규칙"] = top_10_df["조건"] + " ➔ " + top_10_df["결과"]
    
    # 막대그래프 순서를 상위 항목이 위로 오도록 정렬
    top_10_df = top_10_df.sort_values(by="향상도", ascending=True)

    fig = px.bar(
        top_10_df,
        x="향상도",
        y="규칙",
        orientation="h",
        text="향상도",
        color="향상도",
        color_continuous_scale="Viridis",
        labels={"향상도": "향상도 (Lift)", "규칙": "연관 규칙 (조건 ➔ 결과)"}
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(height=400, showlegend=False, yaxis={'categoryorder':'array', 'categoryarray': top_10_df['규칙']})
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("조건에 해당하는 규칙이 없습니다.")

# 규칙 데이터 표 출력
st.subheader("📋 연관 규칙 목록")
st.dataframe(
    filtered_df.style.format({
        "지지도": "{:.4f}",
        "신뢰도": "{:.4f}",
        "향상도": "{:.2f}"
    }),
    use_container_width=True
)
