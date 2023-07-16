import requests
import pandas as pd


QUESTION_INDICATOR = "Question: "
SEPARATOR = "\n "
ANSWER_INDICATOR = "Answer: "


def download_truthfulqa(output_filepath):
    url = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"

    response = requests.get(url)
    response.raise_for_status()

    with open(output_filepath, "wb") as file:
            file.write(response.content)

    print("Data successfully downloaded")


def get_qa_pair_prompts(questions, answers):
    return QUESTION_INDICATOR + questions + SEPARATOR + ANSWER_INDICATOR + answers


def get_question_prompts(questions):
    return QUESTION_INDICATOR + questions + SEPARATOR + ANSWER_INDICATOR


# Given column c1, for each row, split all elements of c1 by ";", then replace
# such column with all combinations of c1
def split_data(df, c1):
    df[c1] = df[c1].str.split(";")
    df = df.explode(c1)
    df[c1] = df[c1].str.lstrip()

    return df


def generate_labeled_qa_pairs(data_raw_path, data_processed_path):
    data = pd.read_csv(data_raw_path)

    # Create positive and negative examples
    data_pos = data[["Question", "Correct Answers"]].copy()
    data_neg = data[["Question", "Incorrect Answers"]].copy()

    data_pos = split_data(data_pos, "Correct Answers")
    data_neg = split_data(data_neg, "Incorrect Answers")

    # Concatenate question and answer, with separator
    data_pos["Full"] = get_qa_pair_prompts(
        data_pos["Question"], data_pos["Correct Answers"])
    data_neg["Full"] = get_qa_pair_prompts(
        data_neg["Question"], data_neg["Incorrect Answers"])

    # Create unified dataframe
    data_pos["Label"] = 1
    data_neg["Label"] = 0

    labeled_qa_pairs = pd.concat((data_pos, data_neg))[["Full", "Label"]]
    labeled_qa_pairs = labeled_qa_pairs.reset_index().drop(
        columns=["index"])  # Fixes index

    labeled_qa_pairs.to_csv(data_processed_path, index=False)
    print("Dataset successfully processed")
