from langchain_voyageai import VoyageAIEmbeddings

from regurag.config import VOYAGE_API_KEY


def main():
    emb = VoyageAIEmbeddings(voyage_api_key=VOYAGE_API_KEY, model="voyage-3")
    print(emb.embed_query("test query"))  # should print a long float list


if __name__ == "__main__":
    main()
