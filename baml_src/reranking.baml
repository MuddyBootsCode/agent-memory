// Post-retrieval re-ranking for multi-database fan-out results

class ResultItem {
  id string @description("Result ID")
  content string @description("The result content or summary")
  source_db string @description("Which database this came from")
  result_type string @description("Type: message, entity, preference, trace")
}

class ScoredResult {
  id string @description("Result ID (pass through from input)")
  relevance float @description("Relevance to the original query, 0.0 to 1.0")
  keep bool @description("True if this result is relevant enough to show to the user")
  reasoning string @description("Brief explanation of relevance assessment")
}

class RerankOutput {
  scored_results ScoredResult[] @description("Results scored and filtered, ordered by relevance descending")
  total_input int @description("Number of results received")
  total_kept int @description("Number of results kept after filtering")
}

function RerankResults(query: string, results: ResultItem[]) -> RerankOutput {
  client Anthropic
  prompt #"
    You are a relevance judge. Score each search result for how well it answers
    the user's original query.

    Rules:
    - Score 0.0-1.0 based on direct relevance to the query
    - Set keep=true only for results with relevance >= 0.4
    - A result that partially answers the query is still relevant
    - A result that is tangentially related but not useful should score low
    - Be aggressive about filtering noise — better to show 3 great results than 10 mediocre ones

    ## Original Query
    {{ query }}

    ## Results to Score
    {% for r in results %}
    ### Result {{ loop.index }} (id: {{ r.id }}, from: {{ r.source_db }}, type: {{ r.result_type }})
    {{ r.content }}
    {% endfor %}

    {{ ctx.output_format }}
  "#
}

test RerankMeetingResults {
  functions [RerankResults]
  args {
    query "What actions came out of Monday's standup?"
    results [
      {
        id "1"
        content "Monday standup: Sarah to finish API docs by Wednesday. Marcus to review PR #42."
        source_db "meetings"
        result_type "message"
      },
      {
        id "2"
        content "Sarah mentioned she prefers morning meetings over afternoon ones."
        source_db "meetings"
        result_type "preference"
      },
      {
        id "3"
        content "The API documentation project milestone is due Friday."
        source_db "projects"
        result_type "message"
      }
    ]
  }
}
