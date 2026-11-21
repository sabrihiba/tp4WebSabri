
package ma.emsi.sabri.tp4websabri.llm;

import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;

import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.transformer.CompressingQueryTransformer;

import dev.langchain4j.service.AiServices;

import jakarta.enterprise.context.Dependent;

import ma.emsi.sabri.tp4websabri.rag.RagService;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

@Dependent
public class LlmClient implements Serializable {

    private static final long serialVersionUID = 1L;

    private String systemRole;
    private Assistant assistant;
    private ChatMemory chatMemory;

    public LlmClient() {

        // 1. Récupérer la clé API Gemini
        String apiKey = System.getenv("GEMINI_KEY");

        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException(
                    "Variable GEMINI_KEY introuvable");
        }

        // 2. Créer le modèle Gemini
        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-3.5-flash-lite")
                .build();

        // 3. Charger et indexer les deux PDF
        RagService ragService = new RagService();

        // 4. Retriever du premier PDF : RAG
        ContentRetriever ragRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(ragService.getRagStore())
                        .embeddingModel(ragService.getEmbeddingModel())
                        .maxResults(3)
                        .build();

        // 5. Retriever du deuxième PDF : LangChain4j
        ContentRetriever langchainRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(ragService.getLangchainStore())
                        .embeddingModel(ragService.getEmbeddingModel())
                        .maxResults(3)
                        .build();

        // 6. Associer chaque retriever à sa description
        Map<ContentRetriever, String> retrievers = new HashMap<>();

        retrievers.put(
                ragRetriever,
                "Support de cours sur le RAG, le fine-tuning, " +
                        "les modèles de langage et les techniques de GenAI."
        );

        retrievers.put(
                langchainRetriever,
                "Support de cours sur LangChain4j, " +
                        "ses composants, ses services IA et son utilisation en Java."
        );

        // 7. Gemini sélectionne le retriever pertinent
        LanguageModelQueryRouter router =
                new LanguageModelQueryRouter(model, retrievers);

        // 8. Transformer les questions de suivi
        CompressingQueryTransformer queryTransformer =
                new CompressingQueryTransformer(model);

        // 9. Construire le pipeline RAG
        DefaultRetrievalAugmentor retrievalAugmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryTransformer(queryTransformer)
                        .queryRouter(router)
                        .build();

        // 10. Créer la mémoire conversationnelle
        chatMemory = MessageWindowChatMemory.withMaxMessages(20);

        // 11. Construire l'assistant avec le RAG
        assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .retrievalAugmentor(retrievalAugmentor)
                .build();
    }

    public void setSystemRole(String systemRole) {

        this.systemRole = systemRole;

        chatMemory.clear();
        chatMemory.add(SystemMessage.from(systemRole));
    }

    public String chat(String prompt) {
        return assistant.chat(prompt);
    }
}
