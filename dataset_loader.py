from datasets import load_dataset

def load_imdb_dataset():
    dataset = load_dataset("imdb")

    train_data = dataset["train"]
    test_data = dataset["test"]

    return train_data, test_data


if __name__ == "__main__":
    train_data, test_data = load_imdb_dataset()

    print("Training Samples:", len(train_data))
    print("Testing Samples:", len(test_data))

    print("\nSample Review:")
    print(train_data[0]["text"])

    print("\nLabel:")
    print(train_data[0]["label"])