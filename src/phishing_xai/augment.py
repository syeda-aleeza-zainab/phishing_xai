"""Synonym replacement data augmentation using WordNet (applied to training set only)."""

import random
from typing import Optional

import pandas as pd
from nltk.corpus import stopwords, wordnet

from phishing_xai.utils import get_logger

logger = get_logger("phishing_xai.augment")


def get_english_stopwords() -> set[str]:
    """Return English stopwords from NLTK."""
    try:
        return set(stopwords.words("english"))
    except LookupError:
        import nltk
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


def synonym_replacement(
    text: str,
    replace_ratio: float = 0.2,
    rng: Optional[random.Random] = None,
) -> str:
    """Replace non-stopword words with WordNet synonyms deterministically.
    
    Preserves original notebook algorithm behavior while injecting a controlled RNG.
    """
    if rng is None:
        rng = random.Random(42)

    stop_words = get_english_stopwords()
    words = text.split()
    if not words:
        return text

    new_words = words[:]
    non_stop_indices = [i for i, w in enumerate(words) if w.lower() not in stop_words]

    if not non_stop_indices:
        return text

    num_replacements = max(1, int(len(non_stop_indices) * replace_ratio))
    indices_to_replace = rng.sample(non_stop_indices, min(num_replacements, len(non_stop_indices)))

    for idx in indices_to_replace:
        word = words[idx]
        synonyms = wordnet.synsets(word)
        valid_synonyms = [
            lemma.name().replace("_", " ")
            for syn in synonyms
            for lemma in syn.lemmas()
            if lemma.name().lower() != word.lower()
        ]

        if valid_synonyms:
            synonym = rng.choice(valid_synonyms)
            new_words[idx] = synonym

    return " ".join(new_words)


def balance_training_data(
    train_df: pd.DataFrame,
    text_col: str = "clean_text",
    label_col: str = "label",
    replace_ratio: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """Balance the training set by generating synonym-augmented samples for the minority (phishing) class.
    
    IMPORTANT: This function must ONLY ever be applied to the training set.
    """
    rng = random.Random(seed)

    phishing_train = train_df[train_df[label_col] == 1].copy()
    safe_train = train_df[train_df[label_col] == 0].copy()

    majority_count = len(safe_train)
    minority_count = len(phishing_train)
    samples_to_generate = majority_count - minority_count

    logger.info(
        f"Balancing training data: Safe={majority_count}, Phishing={minority_count}. "
        f"Augmenting {samples_to_generate} samples for the phishing class."
    )

    if samples_to_generate <= 0:
        logger.info("Training set is already balanced or phishing is majority; no augmentation needed.")
        return train_df.copy().reset_index(drop=True)

    augmented_records = []
    while len(augmented_records) < samples_to_generate:
        for _, row in phishing_train.iterrows():
            aug_text = synonym_replacement(row[text_col], replace_ratio=replace_ratio, rng=rng)
            record = row.to_dict()
            record[text_col] = aug_text
            augmented_records.append(record)
            if len(augmented_records) >= samples_to_generate:
                break

    augmented_df = pd.DataFrame(augmented_records)
    balanced_df = pd.concat([train_df, augmented_df], ignore_index=True)

    # Deterministically shuffle the balanced training set
    balanced_df = balanced_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    logger.info(
        f"Augmentation complete: Final training size={len(balanced_df)} "
        f"(Safe: {(balanced_df[label_col] == 0).sum()}, Phishing: {(balanced_df[label_col] == 1).sum()})."
    )
    return balanced_df
