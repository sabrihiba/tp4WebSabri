
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

import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;

import ma.emsi.sabri.tp4websabri.rag.LoggingContentRetriever;
import ma.emsi.sabri.tp4websabri.rag.RagService;

import java.io.Serializable;
import java.util.LinkedHashMap;
import java.util.Map;

@Dependent
public class LlmClient implements Serializable {

    private static final long serialVersionUID = 1L;

    @Inject
    private transient RagService ragService;

    private String systemRole;
    private transient Assistant assistant;
    private transient ChatMemory chatMemory;

    public LlmClient() {
        // Constructeur requis par CDI
    }

    @PostConstruct
    public void initialize() {

        String apiKey = System.getenv("GEMINI_KEY");

        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException(
                    "Variable GEMINI_KEY introuvable");
        }

        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-3.5-flash-lite")
                .build();

        // Utiliser les index partagés du RagService
        ContentRetriever ragRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(ragService.getRagStore())
                        .embeddingModel(ragService.getEmbeddingModel())
                        .maxResults(3)
                        .build();

        ContentRetriever langchainRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(ragService.getLangchainStore())
                        .embeddingModel(ragService.getEmbeddingModel())
                        .maxResults(3)
                        .build();

        // Logs de routage
        ContentRetriever loggedRagRetriever =
                new LoggingContentRetriever(
                        "rag.pdf", ragRetriever);

        ContentRetriever loggedLangchainRetriever =
                new LoggingContentRetriever(
                        "langchain4j.pdf", langchainRetriever);

        // Description des sources
        Map<ContentRetriever, String> retrievers =
                new LinkedHashMap<>();

        retrievers.put(
                loggedRagRetriever,
                "Support de cours sur le RAG, le fine-tuning, " +
                        "les modèles de langage et les techniques de GenAI."
        );

        retrievers.put(
                loggedLangchainRetriever,
                "Support de cours sur LangChain4j, " +
                        "ses composants, ses services IA et son utilisation en Java."
        );

        LanguageModelQueryRouter router =
                new LanguageModelQueryRouter(model, retrievers);

        CompressingQueryTransformer queryTransformer =
                new CompressingQueryTransformer(model);

        DefaultRetrievalAugmentor retrievalAugmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryTransformer(queryTransformer)
                        .queryRouter(router)
                        .build();

        // Mémoire propre à cette instance
        chatMemory = MessageWindowChatMemory.withMaxMessages(20);

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
