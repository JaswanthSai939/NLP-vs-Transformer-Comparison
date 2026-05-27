traditional_accuracy = 84.70
transformer_accuracy = 82.50

print("\n==============================")
print(" MODEL COMPARISON ")
print("==============================")

print(f"\nTraditional NLP Accuracy : {traditional_accuracy:.2f}%")

print(f"Transformer Accuracy     : {transformer_accuracy:.2f}%")

print("\n==============================")

if traditional_accuracy > transformer_accuracy:

    print("\nBest Model: Traditional NLP Model")

elif transformer_accuracy > traditional_accuracy:

    print("\nBest Model: Transformer Model")

else:

    print("\nBoth models performed equally.")

print("\n==============================")