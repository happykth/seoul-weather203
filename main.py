import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import urllib.request

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(page_title="서울 100년 기온 변화", page_icon="🌡️", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/seoul.csv"


# -----------------------------
# 한글 폰트 설정 (그래프에 한글이 깨지지 않도록)
# -----------------------------
@st.cache_resource
def set_korean_font():
    font_url = "https://raw.githubusercontent.com/google/fonts/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
    font_path = "/tmp/NanumGothic-Regular.ttf"
    try:
        urllib.request.urlretrieve(font_url, font_path)
        fm.fontManager.addfont(font_path)
        font_name = fm.FontProperties(fname=font_path).get_name()
        mpl.rcParams["font.family"] = font_name
    except Exception:
        # 폰트 다운로드에 실패해도 앱은 계속 동작하도록 처리
        mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["axes.unicode_minus"] = False


set_korean_font()


# -----------------------------
# 데이터 불러오기
# -----------------------------
@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv(DATA_URL)
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    return df


@st.cache_data(show_spinner=False)
def compute_yearly(df, min_days=300):
    # 관측일수가 너무 적은 연도(첫해·마지막해 등)는 평균이 왜곡될 수 있어 제외
    yearly = df.groupby("연도").agg(
        평균기온=("평균기온", "mean"),
        최저기온=("최저기온", "mean"),
        최고기온=("최고기온", "mean"),
        관측일수=("평균기온", "count"),
    ).reset_index()
    yearly = yearly[yearly["관측일수"] >= min_days].reset_index(drop=True)
    return yearly


# -----------------------------
# 화면 구성
# -----------------------------
st.title("🌡️ 서울, 100년의 기온 변화")
st.markdown(
    "서울 기상 관측 데이터를 바탕으로, **연평균 기온이 지난 100여 년 동안 어떻게 변해왔는지** 한눈에 살펴봅니다."
)

with st.spinner("데이터를 불러오는 중입니다..."):
    df = load_data()

yearly = compute_yearly(df)

if yearly.empty:
    st.error("데이터를 불러오지 못했습니다. 잠시 후 다시 시도해주세요.")
    st.stop()

start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())

# -----------------------------
# 사이드바: 연도 범위 선택
# -----------------------------
st.sidebar.header("🔎 조회 기간 설정")
year_range = st.sidebar.slider(
    "연도 범위를 선택하세요",
    min_value=start_year,
    max_value=end_year,
    value=(start_year, end_year),
    step=1,
)

view = yearly[(yearly["연도"] >= year_range[0]) & (yearly["연도"] <= year_range[1])]

# -----------------------------
# 핵심 지표 요약
# -----------------------------
first_temp = view.iloc[0]["평균기온"]
last_temp = view.iloc[-1]["평균기온"]
diff = last_temp - first_temp

col1, col2, col3 = st.columns(3)
col1.metric(f"{int(view.iloc[0]['연도'])}년 연평균 기온", f"{first_temp:.1f} ℃")
col2.metric(f"{int(view.iloc[-1]['연도'])}년 연평균 기온", f"{last_temp:.1f} ℃")
col3.metric("변화량", f"{diff:+.1f} ℃")

st.divider()

# -----------------------------
# 메인 그래프: 연평균 기온 추이 + 추세선
# -----------------------------
st.subheader("📈 연평균 기온 추이")

fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(
    view["연도"], view["평균기온"],
    color="#4C72B0", linewidth=1.5, alpha=0.85, label="연평균 기온"
)
ax.scatter(view["연도"], view["평균기온"], color="#4C72B0", s=10, alpha=0.5)

# 추세선 (선형 회귀)
coeffs = np.polyfit(view["연도"], view["평균기온"], 1)
trend = np.poly1d(coeffs)
ax.plot(
    view["연도"], trend(view["연도"]),
    color="#C44E52", linewidth=2.5, linestyle="--",
    label=f"추세선 (10년당 약 {coeffs[0]*10:+.2f} ℃)"
)

ax.set_xlabel("연도")
ax.set_ylabel("연평균 기온 (℃)")
ax.set_title("서울 연평균 기온 변화")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

st.pyplot(fig)

st.caption(
    f"선택한 기간 동안 서울의 연평균 기온은 10년마다 약 **{coeffs[0]*10:+.2f}℃** 씩 "
    f"{'상승' if coeffs[0] > 0 else '하강'}하는 추세를 보입니다."
)

st.divider()

# -----------------------------
# 최고·최저 기온 추이
# -----------------------------
st.subheader("🔺🔻 연평균 최고·최저 기온 추이")

fig2, ax2 = plt.subplots(figsize=(11, 5))
ax2.plot(view["연도"], view["최고기온"], color="#DD8452", linewidth=1.5, label="연평균 최고기온")
ax2.plot(view["연도"], view["평균기온"], color="#4C72B0", linewidth=1.5, label="연평균 기온")
ax2.plot(view["연도"], view["최저기온"], color="#55A868", linewidth=1.5, label="연평균 최저기온")
ax2.set_xlabel("연도")
ax2.set_ylabel("기온 (℃)")
ax2.legend(loc="upper left")
ax2.grid(alpha=0.3)

st.pyplot(fig2)

st.divider()

# -----------------------------
# 일별 평균기온 분포 (히스토그램)
# -----------------------------
st.subheader("📊 일별 평균기온 분포")

daily_view = df[(df["연도"] >= year_range[0]) & (df["연도"] <= year_range[1])]

bin_width = st.slider("구간(bin) 폭 선택 (℃)", min_value=1, max_value=5, value=2, step=1)

min_t = np.floor(daily_view["평균기온"].min())
max_t = np.ceil(daily_view["평균기온"].max())
bins = np.arange(min_t, max_t + bin_width, bin_width)

fig3, ax3 = plt.subplots(figsize=(11, 5))
counts, edges, patches = ax3.hist(
    daily_view["평균기온"], bins=bins, color="#F4B942", edgecolor="white"
)

mean_t = daily_view["평균기온"].mean()
median_t = daily_view["평균기온"].median()
ax3.axvline(mean_t, color="#C44E52", linestyle="--", linewidth=2, label=f"평균 {mean_t:.1f}℃")
ax3.axvline(median_t, color="#4C72B0", linestyle=":", linewidth=2, label=f"중앙값 {median_t:.1f}℃")

ax3.set_xlabel("일별 평균기온 (℃)")
ax3.set_ylabel("일수")
ax3.set_title(f"{year_range[0]}년 ~ {year_range[1]}년 일별 평균기온 분포")
ax3.legend(loc="upper right")
ax3.grid(alpha=0.3)

st.pyplot(fig3)

# 어느 구간에 가장 많은 날이 몰려 있는지 안내
peak_idx = np.argmax(counts)
peak_start, peak_end = edges[peak_idx], edges[peak_idx + 1]
st.caption(
    f"선택한 기간 동안 총 **{len(daily_view):,}일**의 기록 중, "
    f"가장 많은 날이 몰려 있는 구간은 **{peak_start:.0f}℃ ~ {peak_end:.0f}℃**로 "
    f"{int(counts[peak_idx]):,}일({counts[peak_idx]/len(daily_view)*100:.1f}%)이 여기에 해당해요."
)

st.divider()

# -----------------------------
# 원본 데이터 (연도별 요약) 보기
# -----------------------------
with st.expander("📋 연도별 데이터 표로 보기"):
    st.dataframe(
        view.rename(columns={"관측일수": "연간 관측일수"}).round(2),
        use_container_width=True,
        hide_index=True,
    )

st.caption("데이터 출처: 기상청 서울(지점번호 108) 관측 자료")
