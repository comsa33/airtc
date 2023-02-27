import streamlit as st

import functions as funcs


st.session_state.jp_df, st.session_state.comp_name_ls = funcs.get_data()


with st.sidebar:
    st.text('[데이터 필터]')
    comp_nm = st.selectbox(
        "회사명을 입력/선택하세요.",
        st.session_state.comp_name_ls
    )
    max_chunk_size = st.slider(
        'batch size를 선택하세요.',
        0, 2000, (1500)
    )
    sample_n = st.slider(
                "✓ 딥러닝 모델에 추론할 데이터 총 개수를 선택하세요.",
                1, 30, (10)
    )

reviews_all = ' '.join(st.session_state.jp_df.query(f'comp_nm == "{comp_nm}"')['all_text'].tolist())
reviews_chunks = funcs.divide_into_chunks(reviews_all, max_chunk_size)

candidate_labels = ['적극적', '수동적', '자신감', '신중함', '책임감', '무심함', '개인성향', '조직성향', '수평적', '위계적']
multi_label_input = "ON"

df_concat = funcs.get_df_concat(reviews_chunks, candidate_labels, sample_n)

result_dict = {}

for col_no in range(0, 10, 2):
    for i, row in df_concat.iterrows():
        labels = row[f'labels_{col_no}']
        scores = row[f'scores_{col_no}']
        for label, score in zip(labels, scores):
            if label in result_dict:
                result_dict[label].append(score)
            else:
                result_dict[label] = [score]

# Calculate the mean score for each label
for label in result_dict:
    result_dict[label] = sum(result_dict[label])/len(result_dict[label])

categories1 = ['적극적', '자신감', '책임감', '수동적', '신중함', '무심함', '적극적']
values1 = []
for key in categories1:
    values1.append(result_dict[key])

categories2 = ['조직성향', '위계적', '개인성향', '수평적', '조직성향']
values2 = []
for key in categories2:
    values2.append(result_dict[key])

user1 = [0.7, 0.7, 1.0, 0.3, 0.3, 0.0, 0.7]
user2 = [0.9, 0.4, 0.1, 0.6, 0.9]

st.title('[그레이비랩 기업부설 연구소 / AI lab.]')

funcs.draw_radar_chart(values1, categories1, user1, 'AIR', 'Ruo', comp_nm)
funcs.draw_radar_chart(values2, categories2, user2, 'TC', 'Ruo', comp_nm)
