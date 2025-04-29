from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class RAGChain:
    """
    Implements a Retrieval-Augmented Generation (RAG) chain for technical support queries.

    This class builds a processing chain that combines retrieved technical documentation
    and incident records with a language model (using gpt-4o) to generate comprehensive
    solution guides for technical support queries.

    The chain uses a specialized prompt template that can prioritize solution steps from
    technical documentation or incident records based on availability and relevance.
    The resulting output is formatted as a clear, step-by-step solution guide tailored
    to the user's specific question.
    """
    def __init__(self, template_name="tech_incident_template", model_name="gpt-4o"):
        self.template_name = template_name

        self.template = (
            "You are a technical support assistant. Based on the following retrieved documents,"
            " generate a step-by-step guide to help the user solve their issue."
            "\n\n"
            "Technical Documentation:\n{tech_docs}\n\n"
            "Incident Records:\n{incident_docs}\n\n"
            "User Query: {question}\n\n"
            "Provide a clear, concise, and actionable solution guide."
        )

        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        self.output_parser = StrOutputParser()

        self.chain = (
            self.prompt
            | self.llm
            | self.output_parser
        )

    def run(self, query, tech_results, incident_results):
        if not isinstance(query, str):
            raise ValueError("Query must be a string")

        if len(query.strip()) < 5:
            return "Please enter Valid Query"

        try:
            tech_text = "\n".join([doc.page_content for doc in tech_results]) if isinstance(tech_results, list) else tech_results
            incident_text = "\n".join([doc.page_content for doc in incident_results]) if isinstance(incident_results, list) else incident_results

            inputs = {
                "query": query,
                "tech_results": tech_text,
                "incident_results": incident_text
            }

            return self.chain.invoke(inputs)

        except Exception as e:
            return f"Error processing query: {str(e)}"
