
package ma.emsi.sabri.tp4websabri.rag;

import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.query.Query;

import java.util.List;
import java.util.logging.Logger;

public class LoggingContentRetriever implements ContentRetriever {

    private static final Logger LOGGER =
            Logger.getLogger(LoggingContentRetriever.class.getName());

    private final String documentName;
    private final ContentRetriever delegate;

    public LoggingContentRetriever(
            String documentName,
            ContentRetriever delegate) {

        this.documentName = documentName;
        this.delegate = delegate;
    }

    @Override
    public List<Content> retrieve(Query query) {

        LOGGER.info("RAG - PDF sélectionné : " + documentName);
        LOGGER.info("RAG - Requête : " + query.text());

        List<Content> results = delegate.retrieve(query);

        LOGGER.info("RAG - Passages récupérés : " + results.size());

        return results;
    }
}
