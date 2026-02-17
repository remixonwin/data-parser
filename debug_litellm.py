import sys
try:
    import litellm
    print(f"litellm: {litellm}")
    from litellm.router import Router as LiteLLMRouter
    print(f"LiteLLMRouter: {LiteLLMRouter}")
    if LiteLLMRouter is None:
        print("LiteLLMRouter is None!")
    else:
        print(f"LiteLLMRouter is callable: {callable(LiteLLMRouter)}")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Other Error: {e}")
