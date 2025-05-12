# Conversational Anomaly Detection Engine

## 1. Overview

This project explores two different approaches for detecting anomalies in conversations between users and an AI agent (StoryBot), based on the provided `conversations.json` dataset. The goal is to identify unusual patterns in user messages to help maintain a safe and engaging environment.

The two approaches are implemented in separate Jupyter notebooks:

* `anom_det_con_unsuper_clust.ipynb`: Implements an anomaly detection strategy using sentence embeddings, dimensionality reduction (PCA), and K-Means clustering.
* `anom_det_con_FeatureBased.ipynb`: Implements a feature-based anomaly detection strategy using sentiment analysis, message length, and sentiment shifts.

## 2. Dataset

The primary dataset used is `conversations.json`.
* **Format:** A JSON file containing a list of conversations.
* **Structure of each conversation:**
    * `messages_list`: A list of message objects.
        * Each message object includes `ref_conversation_id`, `ref_user_id` (1 for StoryBot), `transaction_datetime_utc`, `screen_name`, and `message` (the text).
    * `ref_conversation_id`: Top-level identifier for the conversation.
    * `ref_user_id`: Top-level identifier for the human user in the conversation.

This file is expected to be in a `data` subdirectory relative to the notebooks (i.e., `./data/conversations.json`).

## 3. Notebook Descriptions

### 3.1. `anom_det_con_unsuper_clust.ipynb` (Embedding & Clustering Approach)

This notebook detects anomalies in user messages based on their semantic meaning.

**Methodology:**
1.  **Data Handling:** Loads `conversations.json` and splits conversations into 80% training and 20% testing sets. User messages are extracted from these sets.
2.  **Sentence Embeddings:** Converts the text of each user message into a numerical vector (embedding) using the `all-MiniLM-L6-v2` Sentence Transformer model.
3.  **Dimensionality Reduction:** Applies Principal Component Analysis (PCA) to the training embeddings to reduce dimensionality while retaining 90% of the variance. The same PCA transformation is applied to test embeddings.
4.  **Optimal `k` Selection:** Determines a suitable number of clusters (`k`) for K-Means by analyzing the Elbow method (WCSS) and Silhouette Scores on the PCA-reduced training embeddings.
5.  **K-Means Clustering:** Fits a K-Means model on the PCA-reduced training embeddings using the chosen `k`.
6.  **Anomaly Detection:**
    * An anomaly threshold is calculated based on the 95th (or 99th, as experimented in the notebook) percentile of distances of training messages to their closest cluster centroids.
    * User messages in the test set whose distance to their nearest cluster centroid exceeds this threshold are flagged as anomalous.
    * Conversations from the test set containing these anomalous messages are identified.

**Limitations:**
* **Small Dataset:** The `conversations.json` file contains a small number of conversations (24 total). Training unsupervised models like K-Means on a limited number of message embeddings (approx. 281 for training after split) can lead to less stable or less generalizable clusters and anomaly thresholds.
* **PCA Information Loss:** While PCA retains a high percentage of variance, some subtle semantic information relevant to specific anomalies might still be lost during dimensionality reduction.
* **K-Means Assumptions:** K-Means tends to find spherical clusters and might struggle with clusters of irregular shapes or varying densities, which can be characteristic of text embedding distributions.
* **Threshold Sensitivity:** The percentile-based threshold for anomaly distance is a heuristic. Its effectiveness depends on the distribution of "normal" data, and it might flag rare but benign messages as anomalies or miss subtle anomalies.
* **Context Window:** This approach analyzes individual user messages for semantic abnormality. It doesn't explicitly model conversational flow or multi-turn context beyond what's implicitly captured in the message embeddings themselves.

**Possible Extensions:**
* **Larger Dataset:** Using a significantly larger dataset for training would improve the robustness of the clusters and anomaly detection.
* **Alternative Anomaly Detection on Embeddings:** Instead of PCA + K-Means, I could explore direct anomaly detection algorithms on the original high-dimensional embeddings, such as Isolation Forest, One-Class SVM, or Autoencoders.
* **Hierarchical Clustering or Density-Based Clustering (DBSCAN):** These could be explored as alternatives to K-Means if non-spherical clusters are suspected.
* **Ensemble Methods:** Combine this embedding-based approach with the feature-based approach for a more robust detection system.
* **Qualitative Analysis of Clusters:** Manually inspect messages within each cluster to understand the "topics" or "types" of user utterances K-Means is identifying, which can help validate the meaningfulness of the clusters.

### 3.2. `anom_det_con_FeatureBased.ipynb` (Feature Engineering Approach)

This notebook detects anomalies based on engineered features from user messages, specifically focusing on message length and sentiment flips.

**Methodology:**
1.  **Data Handling:** Loads `conversations.json`, splits into 80% training and 20% testing sets. User messages are extracted.
2.  **Feature Engineering:** For each user message:
    * `sentiment_compound`: VADER sentiment score.
    * `message_length`: Number of characters.
    * `user_sentiment_shift`: Change in VADER compound score from the user's previous message in the same conversation (calculated within the `engineer_features_for_messages` function).
3.  **Define Anomaly Criteria & Thresholds:**
    * `len_thr_shrt`: Threshold for "too short" messages, derived from the 5th percentile of message lengths in the training data.
    * `sentim_pos_thr` / `sentim_neg_thr`: Thresholds (0.05 / -0.05) to categorize VADER sentiment scores into 'positive', 'negative', or 'neutral'.
4.  **Anomaly Detection (Rule-Based):**
    * For each user message in the test set, two flags are generated (1 for True, 0 for False):
        * `isAnomalousTooShort`: True if `message_length < len_thr_shrt`.
        * `isAnomalousSentimentFlip`: True if the user's sentiment category (positive/negative) flips compared to their immediately preceding message in the same conversation.
    * An `isOverallAnomaly` flag is then set to 1 if *either* `isAnomalousTooShort` OR `isAnomalousSentimentFlip` is 1.
5.  **Output:**
    * Creates a pandas DataFrame (`df_test_message_analysis`) with test user messages, their engineered features, and anomaly flags.
    * Exports this DataFrame to `test_message_anomaly_analysis.csv`.
    * Identifies and lists conversations from the test set that contain messages flagged with `isOverallAnomaly`.

**Limitations:**
* **Rule-Based Rigidity:** The current rules are specific (too short OR sentiment flip). This might miss other types of anomalies (e.g., unusual topics, keyword usage not yet implemented, complex behavioral patterns).
* **VADER Sentiment Nuance:** VADER is good for general sentiment but can miss context or sarcasm. A detected "negative" sentiment or a "flip" might be contextually appropriate and not a true problematic anomaly (as discussed with the code  example).
* **Threshold Sensitivity:** The `len_thr_shrt` percentile and fixed sentiment category thresholds can impact what's flagged. They might need tuning based on qualitative review.
* **Limited Feature Set:** Currently focuses on length and sentiment flips. Other potentially anomalous behaviors are not yet captured. for example, The response "I don't know" from the chatbot itself could be falegged  as anomaly etc.
* **"Sentiment Flip" Definition:** The current flip detection is based on categorical changes (positive <-> negative). It doesn't account for the *magnitude* of the shift or flips involving neutral states (e.g., positive -> neutral -> negative).

**Possible Extensions:**
* **Expand Feature Set:**
    * Incorporate `user_sentiment_shift` magnitude directly into rules (e.g., flag very large negative shifts).
    * Add keyword spotting for concerning terms or prompt injection patterns.
    * Include interaction features: user/bot response times, message frequency
    * Analyze message structure: use of excessive punctuation, ALL CAPS, question rates, using emojis etc
* **More Sophisticated Rules:** Develop more complex conditional rules (e.g., a sentiment flip is only anomalous if the resulting sentiment is *very* negative and the shift magnitude is large).
* **Sequential Anomaly Detection:** For more robust detection of "sudden changes," explore techniques that explicitly model sequences (e.g., looking at a rolling window of features, or more advanced time-series anomaly detection methods if data volume allows).
* **Combined Scoring:** Develop a scoring system where different anomalous feature values contribute to an overall anomaly score for a message, rather than just binary flags.
* **Human-in-the-Loop:** Implement a mechanism to review flagged anomalies and use this feedback to refine rules and thresholds.

## 4. How to Run

1.  **Environment:** Ensure you have Python and the necessary libraries (pandas, numpy, scikit-learn, matplotlib, jupyter, sentence-transformers, vaderSentiment) installed.
2.  **Data:** Place `conversations.json` in a `./data/` subdirectory.
3.  **Jupyter:** Start Jupyter Lab or Jupyter Notebook in the project's root directory.
4.  **Execution:** Open the desired notebook (`.ipynb` file) and run the cells sequentially 
    * For `anom_det_con_unsuper_clust.ipynb`, after viewing the diagnostic plots, change  pre-set value for final cluster value based on previous analysis (e.g., k=11).

## 5. Output / Results

* **Console Output:** Both notebooks print progress, data shapes, derived thresholds, and summaries of detected anomalies/flagged conversations directly in the notebook cells.
* **Plots (`anom_det_con_unsuper_clust.ipynb`):** Generates plots for WCSS (Elbow method) and Silhouette Scores to aid in `k` selection for K-Means.
* **CSV File (`anom_det_con_FeatureBased.ipynb`):** Creates `test_message_anomaly_analysis.csv` containing test user messages with their engineered features and anomaly flags.
* **Flagged Conversations:** Both notebooks will output (in the console) a list of conversation IDs (and potentially other details) from the test set that were identified as containing anomalous user messages according to their respective methodologies.

## 6. "Testing" / Verification

Verification for these notebooks involves:
1.  **Code Execution:** Confirming the notebooks run end-to-end without errors.
2.  **Output Inspection:** Reviewing the console outputs, generated plots, and the CSV file to understand the results.
3.  **Qualitative Analysis of Anomalies:** Manually examine the specific messages and conversations flagged as anomalous.
    * Does the definition of "anomaly" used by the notebook make sense for these examples?
    * Are there false positives (normal messages flagged as anomalous)?
    * Are there potential false negatives (obvious anomalies that were missed)?
4.  **Threshold/Parameter  Changes:** (For further exploration) Experiment by changing thresholds (e.g., percentile for anomaly distance, message length, sentiment shift magnitude) or model parameters (e.g., `k` in K-Means) to observe the impact on detection results.

## 7. Real time adaptaion
While the current implementation processes messages in batches for demonstration and evaluation, the core pipeline is designed so that **real-time anomaly detection can be achieved with minimal changes**:
for example, The PCA and K-Means models are trained offline on historical data. Once trained, these models can be saved and reused for inference on new messages as they arrive, without retraining.( e.g `joblib.dump(pca, 'pca_model.joblib'))
- **Per-Message Processing:**  
  Each new incoming message can be processed individually:
    - Convert the message to an embedding using the same Sentence Transformer model.
    - Apply the saved PCA transformation to reduce dimensionality.
    - Assign the message to a cluster using the saved K-Means model.
    - Compute the distance to the nearest cluster centroid.
    - Flag the message as anomalous if its distance exceeds the pre-computed anomaly threshold.(can be sent to through cloud logging solution (e.g., AWS CloudWatch, Azure Monitor, Google Cloud Logging)