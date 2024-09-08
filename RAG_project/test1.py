from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 예제 문서와 쿼리
documents = [
    "마이크로서비스는 작은 서비스로 나눠진 아키텍처입니다.",
    "단일 애플리케이션을 여러 서비스로 나누면 확장성과 유연성이 높아집니다.",
    "마이크로서비스의 장점은 독립적인 배포와 확장이 가능합니다.",
    "로드 밸런싱은 트래픽을 여러 서버로 분산시킵니다."
]
query = "마이크로서비스의 장점"

# 1. 초기 검색: TF-IDF 벡터화
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)
query_vector = vectorizer.transform([query])

# 2. Re-ranking: 코사인 유사도 계산
cosine_similarities = cosine_similarity(query_vector, doc_vectors).flatten()

# 3. 결과 정렬
ranked_doc_indices = cosine_similarities.argsort()[::-1]
ranked_documents = [documents[i] for i in ranked_doc_indices]

# 출력 결과
print("Re-ranked Documents:")
for i, doc in enumerate(ranked_documents):
    print(f"{i+1}: {doc}")