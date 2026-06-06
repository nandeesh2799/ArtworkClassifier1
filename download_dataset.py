from __future__ import annotations

from artwork_classifier.data import download_primary_dataset


def main() -> None:
    path = download_primary_dataset()
    print("Path:", path)


if __name__ == "__main__":
    main()

