results = {}

for name, model in models.items():
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    results[name] = acc

    print(name, "Accuracy:", acc)
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

predictions = best_model.predict(X_test)

print("Best Model:", best_model_name)
print(classification_report(y_test, predictions))
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6,5))
plt.imshow(cm, cmap='Blues')

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()
plt.figure(figsize=(8,5))
plt.plot(list(results.keys()), list(results.values()), marker='o')
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.show()