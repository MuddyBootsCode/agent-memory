// Retry policy for transient failures
retry_policy StandardRetry {
  max_retries 2
}

// --- Individual Providers ---

client<llm> OpenAI {
  provider openai
  retry_policy StandardRetry
  options {
    model "gpt-4o-mini"
    temperature 0
  }
}

client<llm> Anthropic {
  provider anthropic
  retry_policy StandardRetry
  options {
    model "claude-sonnet-4-20250514"
    temperature 0
    max_tokens 4096
  }
}

client<llm> Gemini {
  provider google-ai
  retry_policy StandardRetry
  options {
    model "gemini-2.5-flash"
    api_key env.GEMINI_API_KEY
  }
}

// --- Fallback Chain ---
// Tries each provider in order until one succeeds

client<llm> Resilient {
  provider fallback
  options {
    strategy [
      OpenAI
      Anthropic
      Gemini
    ]
  }
}
