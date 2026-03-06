// Database vertical classification for multi-DB routing

enum QueryVertical {
  MEETINGS @alias("k1")
    @description("Meetings, calendars, scheduling, attendees, agendas, standups, syncs, 1:1s, action items from meetings, meeting notes")
  PROJECTS @alias("k2")
    @description("Projects, tasks, milestones, deliverables, sprints, backlogs, dependencies, team assignments, project status, deadlines")
  RESEARCH @alias("k3")
    @description("Research notes, papers, findings, knowledge base, citations, experiments, hypotheses, literature, sources, analysis results")
  GENERAL @alias("k4")
    @description("General memory, personal preferences, facts, people, organizations, locations, conversation history, reasoning traces, anything that doesn't clearly fit other categories")
}

class RoutingTarget {
  vertical QueryVertical @description("The database vertical to query")
  confidence float @description("Confidence this vertical is relevant, 0.0 to 1.0")
  reasoning string @description("Brief explanation of why this vertical is relevant")
}

class RoutingDecision {
  targets RoutingTarget[] @description("Verticals to query, ordered by confidence descending. Only include verticals with confidence > 0.3")
  primary_vertical QueryVertical @description("The single most relevant vertical")
  requires_fanout bool @description("True if multiple verticals must be queried to fully answer the query")
  ambiguous bool @description("True if the query is too vague to route confidently. Set true when no vertical has confidence > 0.6 or the query lacks specificity")
  disambiguation_options string[]? @description("If ambiguous, provide 2-3 concrete interpretations of what the user might mean, each mentioning which vertical it would search. Only populate when ambiguous is true.")
}

function RouteQuery(query: string, context: string?) -> RoutingDecision {
  client Anthropic
  prompt #"
    You are a query router for a knowledge graph with multiple database verticals.
    Analyze the user query and determine which database(s) should be searched.

    Rules:
    - Set requires_fanout to true ONLY when the query genuinely spans multiple domains
    - Confidence reflects how certain you are the vertical contains relevant data
    - Only include verticals with confidence > 0.3
    - GENERAL is the fallback when no specific vertical matches
    - Most queries should target 1-2 verticals, rarely more
    - Set ambiguous to true when the query is vague, lacks context, or could
      reasonably apply to multiple unrelated verticals (e.g., "what did we talk about?",
      "anything new?", "what did Sarah say?")
    - When ambiguous, provide 2-3 specific interpretations the user might mean,
      each mentioning the vertical it would search

    {% if context %}
    ## Recent Context
    {{ context }}
    {% endif %}

    {{ ctx.output_format }}

    {{ _.role("user") }}

    {{ query }}
  "#
}

function RouteStorage(content: string, memory_type: string, context: string?) -> RoutingDecision {
  client Anthropic
  prompt #"
    You are a storage router for a knowledge graph. Determine which database
    vertical should store this content.

    For storage, pick ONE primary vertical. Only set requires_fanout to true
    if the content genuinely belongs in multiple databases (rare).

    Memory type being stored: {{ memory_type }}

    {% if context %}
    ## Context
    {{ context }}
    {% endif %}

    {{ ctx.output_format }}

    {{ _.role("user") }}

    {{ content }}
  "#
}

test MeetingQuery {
  functions [RouteQuery]
  args {
    query "What was discussed in yesterday's standup?"
    context null
  }
}

test CrossDomainQuery {
  functions [RouteQuery]
  args {
    query "What research findings were presented in the project kickoff meeting?"
    context null
  }
}

test ProjectStorage {
  functions [RouteStorage]
  args {
    content "The migration project milestone 2 is delayed by a week due to API dependency"
    memory_type "message"
    context null
  }
}

test AmbiguousQuery {
  functions [RouteQuery]
  args {
    query "What did Sarah say?"
    context "Sarah presented research findings at Monday's team meeting about the migration project"
  }
}

test VagueQuery {
  functions [RouteQuery]
  args {
    query "anything new?"
    context null
  }
}
