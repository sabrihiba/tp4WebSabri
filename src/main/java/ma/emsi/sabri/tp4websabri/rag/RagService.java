
package ma.emsi.sabri.tp4websabri.rag;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;

import java.io.InputStream;

public class RagService {

    private final EmbeddingModel embeddingModel;
    private final EmbeddingStore<TextSegment> ragStore;
    private final EmbeddingStore<TextSegment> langchainStore;

    public RagService() {

        // 1. Créer le modèle d'embeddings local
        embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // 2. Créer un magasin d'embeddings pour chaque PDF
        ragStore = new InMemoryEmbeddingStore<>();
        langchainStore = new InMemoryEmbeddingStore<>();

        // 3. Charger et indexer les deux documents
        indexPdf("documents/rag.pdf", ragStore);
        indexPdf("documents/langchain4j.pdf", langchainStore);
    }

    private void indexPdf(
            String resourcePath,
            EmbeddingStore<TextSegment> store) {

        // Charger le PDF depuis src/main/resources
        InputStream input = RagService.class
                .getClassLoader()
                .getResourceAsStream(resourcePath);

        if (input == null) {
            throw new IllegalArgumentException(
                    "PDF introuvable : " + resourcePath);
        }

        try (InputStream stream = input) {

            // Extraire le texte du PDF
            DocumentParser parser = new ApacheTikaDocumentParser();
            Document document = parser.parse(stream);

            // Découper, calculer les embeddings et les stocker
            EmbeddingStoreIngestor ingestor =
                    EmbeddingStoreIngestor.builder()
                            .documentSplitter(
                                    DocumentSplitters.recursive(1000, 150))
                            .embeddingModel(embeddingModel)
                            .embeddingStore(store)
                            .build();

            ingestor.ingest(document);

            System.out.println(
                    "PDF indexé : " + resourcePath);

        } catch (Exception e) {
            throw new RuntimeException(
                    "Erreur d'indexation : " + resourcePath, e);
        }
    }

    public EmbeddingModel getEmbeddingModel() {
        return embeddingModel;
    }

    public EmbeddingStore<TextSegment> getRagStore() {
        return ragStore;
    }

    public EmbeddingStore<TextSegment> getLangchainStore() {
        return langchainStore;
    }
}
