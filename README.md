# DDoS Attack Detection Project

This project aims to detect Distributed Denial of Service (DDoS) attacks using various machine learning techniques. The dataset contains network traffic data, and the goal is to classify the traffic as either benign or DDoS.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Feature Selection and Dimensionality Reduction](#feature-selection-and-dimensionality-reduction)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Neural Network Training](#neural-network-training)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/yourprojectname.git
    ```
2. Navigate to the project directory:
    ```sh
    cd yourprojectname
    ```
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Load the dataset:
    ```python
    df = pd.read_csv("path/to/your/dataset.csv")
    ```

2. Preprocess the data and visualize:
    ```python
    # Check for missing values
    missing_values = df.isnull().sum()
    print("Missing Values:\n", missing_values)

    # Drop rows with missing values
    df = df.dropna()

    # Summary statistics
    print("\nSummary Statistics:")
    print(df.describe())

    # Visualize numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x=col)
        plt.title(f'Box Plot of {col}')
        plt.show()
    ```

3. Encode categorical features:
    ```python
    label_encoder = LabelEncoder()
    df[' Source IP'] = label_encoder.fit_transform(df[' Source IP'])
    df[' Destination IP'] = label_encoder.fit_transform(df[' Destination IP'])
    df[' Label'] = label_encoder.fit_transform(df[' Label'])
    ```

## Project Structure

- `data/` - Contains the dataset files.
- `notebooks/` - Jupyter notebooks for data exploration and model training.
- `src/` - Source code for data preprocessing, feature selection, model training, and evaluation.
- `models/` - Saved models and evaluation metrics.

## Data Preprocessing

The data preprocessing steps include:

1. Loading the dataset.
2. Checking for and handling missing values.
3. Encoding categorical features using `LabelEncoder`.
4. Visualizing numeric columns with box plots to identify outliers.
5. Generating a correlation matrix and heatmap.

## Feature Selection and Dimensionality Reduction

1. **Random Undersampling:**
    ```python
    rus = RandomUnderSampler(random_state=42)
    X_downsampled, y_downsampled = rus.fit_resample(X, y)
    ```

2. **Standard Scaling:**
    ```python
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.select_dtypes(include=[float, int]))
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    ```

3. **Variance Threshold:**
    ```python
    selector = VarianceThreshold(threshold=0)
    selector.fit(X_scaled_df)
    non_constant_indices = selector.get_support(indices=True)
    X_non_constant = X_scaled_df.iloc[:, non_constant_indices]
    ```

4. **SelectKBest:**
    ```python
    k_best = SelectKBest(score_func=f_classif, k=40)
    X_new = k_best.fit_transform(X_non_constant, y)
    selected_feature_names = X_non_constant.columns[k_best.get_support()]
    X_selected = pd.DataFrame(X_new, columns=selected_feature_names)
    ```

5. **PCA:**
    ```python
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_selected)
    X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(10)])
    ```

## Model Training and Evaluation

Various classifiers are trained and evaluated using accuracy and classification reports:

1. **Classifiers:**
    ```python
    classifiers = {
        'Random Forest': RandomForestClassifier(),
        'XGBoost': XGBClassifier(),
        'Logistic Regression': LogisticRegression(),
        'Support Vector Machine': SVC(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Naive Bayes': GaussianNB(),
        'Gradient Boosting': GradientBoostingClassifier()
    }
    ```

2. **Training and Evaluation:**
    ```python
    for name, classifier in classifiers.items():
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print(f"Classifier: {name}")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Classification Report:\n{report}\n")
    ```

3. **Cross-Validation:**
    ```python
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, classifier in classifiers.items():
        scores = cross_val_score(classifier, X_pca_df, y, cv=cv, scoring='accuracy')
        print(f"Classifier: {name}")
        print(f"Mean Accuracy: {scores.mean():.2f}")
        print(f"Accuracy Std Dev: {scores.std():.2f}")
        print(f"Accuracy Scores: {scores}\n")
    ```

## Neural Network Training

1. **Model Definition:**
    ```python
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(64, activation='relu'),
        Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    ```

2. **Model Compilation:**
    ```python
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    ```

3. **Training with Early Stopping:**
    ```python
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
    ```

4. **Model Evaluation:**
    ```python
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Classification Report:\n{report}")
    ```

5. **Plotting Training History:**
    ```python
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()
    ```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss improvements or bug fixes.

## License

This project is licensed under the MIT License.
