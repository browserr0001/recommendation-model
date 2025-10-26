# Evaluation Metrics Documentation

##  1. Precision@K

**Definition:**  
Proportion of top-K recommended movies that the user actually watched or rated.

**Formula:**  
\[
Precision@K_u = \frac{|recommended_u \cap actual_u|}{K}
\]
\[
Precision@K = \frac{1}{|U|}\sum_{u\in U}Precision@K_u
\]

**Averaging:** Per-User → Mean across all evaluated users  
**Interpretation:**  
If Precision@20 = 0.14, then on average 14% of recommended movies per user were actually consumed.

---

##  2. Recall@K

**Definition:**  
Proportion of all movies a user actually watched that appear in their top-K recommendations.

**Formula:**  
\[
Recall@K_u = \frac{|recommended_u \cap actual_u|}{|actual_u|}
\]
\[
Recall@K = \frac{1}{|U|}\sum_{u\in U}Recall@K_u
\]

**Averaging:** Per-User → Mean  
**Interpretation:**  
If Recall@20 = 0.012, then on average 1.2% of each user’s watched movies were successfully recommended.

---

##  3. NDCG@K (Normalized Discounted Cumulative Gain)

**Definition:**  
Captures **ranking quality** — rewards placing relevant movies near the top of the recommendation list.

**Formula:**  
\[
DCG_u = \sum_{i=1}^{K} \frac{I(item_i\in actual_u)}{\log_2(i+1)}, \quad
NDCG@K_u = \frac{DCG_u}{IDCG_u}
\]
\[
NDCG@K = \frac{1}{|U|}\sum_{u\in U}NDCG@K_u
\]

**Averaging:** Per-User → Mean  
**Interpretation:**  
If NDCG@20 = 0.15, relevant items are generally ranked toward the top of the list.

---

##  4. Hit Rate@K

**Definition:**  
Binary indicator — 1 if a user had at least one relevant item in their top-K recommendations, 0 otherwise.

**Formula:**  
\[
Hit@K_u =
\begin{cases}
1, & \text{if }|recommended_u\cap actual_u|>0\\
0, & \text{otherwise}
\end{cases}
\]
\[
HitRate@K = \frac{1}{|U|}\sum_{u\in U}Hit@K_u
\]

**Averaging:** Per-User → Mean  
**Interpretation:**  
If Hit Rate = 0.18, then 18% of users received at least one good recommendation.

---

##  5. Click-Through Rate (CTR)

**Definition:**  
Fraction of recommendations that resulted in a watch event within a 30-minute window.

**Formula:**  
\[
CTR = \frac{\text{# recs with watch within 30 min}}{\text{# recs served}}
\]

**Averaging:** Global (over all recommendation events)  
**Interpretation:**  
CTR = 0.009 → 0.9% of recommendations triggered immediate engagement.

---

##  6. Diversity

**Definition:**  
Measures how varied the recommendations are — the proportion of unique items (or genres) among all recommendations.

**Formula:**  
\[
Diversity = \frac{|unique\ recommended\ movies|}{|all\ recommended\ movies|}
\]

**Averaging:** Global  
**Interpretation:**  
High diversity (> 0.8) → recommendations span many different titles and genres.

---

##  7. Coverage

**Definition:**  
Proportion of the entire catalog that has been recommended at least once.

**Formula:**  
\[
Coverage = \frac{|unique\ recommended\ movies|}{|total\ catalog\ movies|}
\]

**Averaging:** Global  
**Interpretation:**  
Coverage = 0.37 → the system has recommended 37% of all available movies.

---

##  8. Average Rating

**Definition:**  
Average user rating observed in live rating events (streamed via Kafka).

**Formula:**  
\[
AvgRating = \frac{\sum_i rating_i}{N_{ratings}}
\]

**Averaging:** Global  
**Interpretation:**  
Reflects overall user satisfaction — higher values indicate positive sentiment.

---

## 9. Users Tracked

**Definition:**  
Number of unique users who generated at least one watch event in the current session.

**Formula:**  
\[
UsersTracked = |\{u : u\ \text{has any watch event}\}|
\]

**Averaging:** Count (absolute number)  
**Interpretation:**  
Indicates the scale of active user coverage (e.g., 46k users tracked live).

---

## Summary Table

| Metric | Averaged Over | Type | Captures What |
|:--|:--|:--|:--|
| **Precision@K** | Users | Accuracy | How many recommended movies were actually watched |
| **Recall@K** | Users | Accuracy | How much of each user’s watched set was recovered |
| **NDCG@K** | Users | Ranking Quality | Relevance ordering quality |
| **Hit Rate@K** | Users | Accuracy | Users with ≥ 1 hit |
| **CTR** | Global | Engagement | Immediate interaction rate |
| **Diversity** | Global | Catalog Variety | Spread of unique titles recommended |
| **Coverage** | Global | Catalog Breadth | Portion of catalog recommended |
| **Avg Rating** | Global | Sentiment | Overall user satisfaction |
| **Users Tracked** | Count | Scale | Active user coverage |

---

### Notes

- **Precision, Recall, NDCG, Hit-Rate** → averaged across all users who received recommendations.  
- **CTR, Diversity, Coverage, Avg Rating** → global averages over all events observed.  
- **Users Tracked** → absolute count (not an average).  
- Metrics are computed **continuously** and written to:  
  `metrics/online_evaluation_live.json`  

---

