
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

import jakarta.enterprise.context.ApplicationScoped;

import java.io.InputStream;
import java.util.logging.Logger;

@ApplicationScoped
public class RagService {

    private static final Logger LOGGER =
            Logger.getLogger(RagService.class.getName());

    private final EmbeddingModel embeddingModel;
    private final EmbeddingStore<TextSegment> ragStore;
    private final EmbeddingStore<TextSegment> langchainStore;

    public RagService() {

        LOGGER.info("Initialisation du service RAG");

        embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        ragStore = new InMemoryEmbeddingStore<>();
        langchainStore = new InMemoryEmbeddingStore<>();

        indexPdf("documents/rag.pdf", ragStore);
        indexPdf("documents/langchain4j.pdf", langchainStore);

        LOGGER.info("Indexation des deux PDF terminée");
    }

    private void indexPdf(
            String resourcePath,
            EmbeddingStore<TextSegment> store) {

        InputStream input = Thread.currentThread()
                .getContextClassLoader()
                .getResourceAsStream(resourcePath);

        if (input == null) {
            throw new IllegalArgumentException(
                    "PDF introuvable : " + resourcePath);
        }

        try (InputStream stream = input) {

            DocumentParser parser = new ApacheTikaDocumentParser();
            Document document = parser.parse(stream);

            EmbeddingStoreIngestor ingestor =
                    EmbeddingStoreIngestor.builder()
                            .documentSplitter(
                                    DocumentSplitters.recursive(1000, 150))
                            .embeddingModel(embeddingModel)
                            .embeddingStore(store)
                            .build();

            ingestor.ingest(document);

            LOGGER.info("PDF indexé : " + resourcePath);

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
