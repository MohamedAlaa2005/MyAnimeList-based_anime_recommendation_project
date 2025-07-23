import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.sparse import hstack
from scipy import sparse
def recommend_anime(title, top_n=50):
    animeDf = pd.read_csv('data/anime_genre.csv')
    genre_cols = [c for c in animeDf.select_dtypes(include=["int64","float64"]).columns
                   if set(animeDf[c].dropna().unique()).issubset({0,1})]
    animeDf['synopsis'] = animeDf['synopsis'].fillna("")
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(animeDf['synopsis'])
    genre_scaled = MinMaxScaler().fit_transform(animeDf[genre_cols])
    combined_features = hstack([genre_scaled, tfidf_matrix])
    cos_sim_matrix = cosine_similarity(combined_features)
    index = animeDf[animeDf['title'] == title].index
    if index.empty:
        return f"No anime found with title: {title}"
    idx = index[0]
    sim_scores = list(enumerate(cos_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    results = animeDf.iloc[[i[0] for i in sim_scores]][['title', 'anime_year', 'score','img_url']]
    results['anime_year']= results['anime_year']
    return results.to_dict(orient='records')

def hybrid_recommend_for_user(user_id, k=10, alpha=0.5):
    animeDf = pd.read_csv(r'data/anime_genre.csv')
    rateDf  = pd.read_csv(r'data/user_rate.csv')
    ratings_df = rateDf[["profile", "anime_uid", "score"]].dropna()
    user_to_idx  = {u: i for i, u in enumerate(ratings_df.profile.unique())}
    item_to_idx  = {a: j for j, a in enumerate(animeDf.uid.unique())}
    idx_to_item  = {j: a for a, j in item_to_idx.items()}

    rows = ratings_df.profile.map(user_to_idx).values
    cols = ratings_df.anime_uid.map(item_to_idx).values
    data = ratings_df.score.values.astype(np.float32)

    R = sparse.csr_matrix((data, (rows, cols)),
                          shape=(len(user_to_idx), len(item_to_idx)))

    binary_cols = [c for c in animeDf.select_dtypes(include=["int64","float64"]).columns
                   if set(animeDf[c].dropna().unique()).issubset({0,1})]

    C = sparse.csr_matrix(animeDf[binary_cols].astype(np.float32).values)

    # Step 2: Helper functions
    def cbf_scores(user_idx):
        start, end = R.indptr[user_idx], R.indptr[user_idx + 1]
        item_idx   = R.indices[start:end]
        ratings    = R.data[start:end]

        if len(item_idx) == 0:
            return np.zeros(C.shape[0], dtype=np.float32)

        pref_vec = C[item_idx].T.dot(ratings)
        pref_vec /= ratings.sum()
        pref_vec = pref_vec.reshape(1, -1)

        sims = cosine_similarity(pref_vec, C).ravel()
        return sims

    def cf_scores(user_idx):
        sims = cosine_similarity(R[user_idx], R).ravel()
        numer = R.T.dot(sims)
        denom = sims.sum() - sims[user_idx]
        preds = numer / denom if denom > 0 else np.zeros_like(numer)
        return preds
    if user_id not in user_to_idx:
        raise ValueError(f"User ID '{user_id}' not found in dataset.")

    uidx = user_to_idx[user_id]
    cbf = cbf_scores(uidx)
    cf  = cf_scores(uidx)

    rated_mask = np.zeros(C.shape[0], dtype=bool)
    rated_mask[R.indices[R.indptr[uidx]:R.indptr[uidx+1]]] = True

    hybrid = alpha * cbf + (1 - alpha) * cf
    hybrid[rated_mask] = -np.inf

    top_idx = np.argpartition(-hybrid, k)[:k]
    top_idx = top_idx[np.argsort(-hybrid[top_idx])]

    result = pd.DataFrame({
        "anime_uid": [idx_to_item[i] for i in top_idx],
        "title":     animeDf.set_index("uid").loc[[idx_to_item[i] for i in top_idx], "title"].values,
        "hybrid_score": hybrid[top_idx].round(3)
    }).reset_index(drop=True)

    return result

# def collaborative_filtering(profile_id, top_n=5):
#     rateDf=pd.read_csv(r"data/user_rate.csv")
#     def build_sparse_matrix(df):
#         user_codes, user_index = pd.factorize(df['profile'])
#         item_codes, item_index = pd.factorize(df['title'])

#         sparse_matrix = sparse.csr_matrix(
#             (df['score'], (user_codes, item_codes)),
#             shape=(len(user_index), len(item_index))
#         )

#         return sparse_matrix, user_index, item_index
#     def preprocess_ratings(df):
#         return df.groupby(['profile', 'title'], as_index=False)['score'].mean()
#     df_clean = preprocess_ratings(rateDf)
#     sparse_matrix, user_index, item_index = build_sparse_matrix(df_clean)

#     if profile_id not in user_index:
#         return f"Profile {profile_id} not found."

#     user_idx = np.where(user_index == profile_id)[0][0]

#     target_vector = sparse_matrix[user_idx]
#     sim_scores = cosine_similarity(target_vector, sparse_matrix).flatten()
#     sim_scores[user_idx] = 0 


#     top_users_idx = np.argsort(sim_scores)[::-1]
#     top_users_sims = sim_scores[top_users_idx]

#     user_rated = target_vector.toarray().flatten()
#     unrated_items_idx = np.where(user_rated == 0)[0]

#     recommendations = []
#     for item_idx in unrated_items_idx:
#         item_ratings = sparse_matrix[top_users_idx, item_idx].toarray().flatten()
#         mask = item_ratings > 0

#         if mask.sum() == 0:
#             continue

#         weighted_sum = np.dot(item_ratings[mask], top_users_sims[mask])
#         sim_total = top_users_sims[mask].sum()

#         predicted = weighted_sum / sim_total if sim_total != 0 else 0
#         recommendations.append((item_index[item_idx],predicted))

#     if not recommendations:
#         return f"No recommendations for profile {profile_id}."

#     rec_df = pd.DataFrame(recommendations, columns=['title', 'predicted_score'])
#     return rec_df.sort_values(by='predicted_score', ascending=False).head(top_n).reset_index(drop=True)
def collaborative_filtering(profile_id, top_n=5):
    rateDf = pd.read_csv(r"data/user_rate.csv")

    def preprocess_ratings(df):
        return df.groupby(['profile', 'title'], as_index=False)['score'].mean()

    def build_sparse_matrix(df):
        user_codes, user_index = pd.factorize(df['profile'])
        item_codes, item_index = pd.factorize(df['title'])

        sparse_matrix = sparse.csr_matrix(
            (df['score'], (user_codes, item_codes)),
            shape=(len(user_index), len(item_index))
        )

        return sparse_matrix, user_index, item_index

    df_clean = preprocess_ratings(rateDf)
    sparse_matrix, user_index, item_index = build_sparse_matrix(df_clean)

    if profile_id not in user_index:
        return f"Profile {profile_id} not found."

    user_idx = np.where(user_index == profile_id)[0][0]

    # Step 1: Compute cosine similarity between user and all others
    sim_scores = cosine_similarity(sparse_matrix[user_idx], sparse_matrix).flatten()
    sim_scores[user_idx] = 0  # exclude self

    # Step 2: Get top similar users (can cap if needed)
    top_users_idx = np.where(sim_scores > 0)[0]
    top_users_sims = sim_scores[top_users_idx]

    # Step 3: Mask rated items
    user_rated_mask = sparse_matrix[user_idx].toarray().ravel() > 0
    unrated_items_idx = np.where(~user_rated_mask)[0]

    # Step 4: Get submatrix of top users’ ratings for unrated items
    sub_matrix = sparse_matrix[top_users_idx][:, unrated_items_idx]  # shape: (top_users, items)

    # Step 5: Weighted prediction using dot product
    weighted_ratings = sub_matrix.multiply(top_users_sims[:, np.newaxis])
    sim_sum = np.maximum(top_users_sims.sum(), 1e-6)  # avoid divide by 0
    predicted_scores = weighted_ratings.sum(axis=0) / sim_sum  # shape: (1, num_unrated_items)

    predicted_scores = np.array(predicted_scores).flatten()

    if len(predicted_scores) == 0:
        return f"No recommendations for profile {profile_id}."

    top_k_indices = np.argsort(-predicted_scores)[:top_n]
    recommended_items = [item_index[unrated_items_idx[i]] for i in top_k_indices]

    rec_df = pd.DataFrame({
        'title': recommended_items,
        'predicted_score': predicted_scores[top_k_indices].round(3)
    })

    return rec_df.reset_index(drop=True)