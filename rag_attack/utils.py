def test_connection():
    """
    Dummy function to test package import
    """
    return "RAG Attack package successfully loaded!"

def validate_config(config):
    """
    Validate configuration dictionary
    """
    required_keys = [
        'search_endpoint', 'search_key', 'sql_server', 
        'sql_database', 'sql_username', 'sql_password',
        'api_base_url', 'openai_endpoint', 'openai_key', 
        'chat_deployment'
    ]
    
    missing_keys = [key for key in required_keys if key not in config or not config[key]]
    
    if missing_keys:
        return False, f"Missing configuration keys: {', '.join(missing_keys)}"
    return True, "Configuration is valid"