# -*- coding: utf-8 -*-

import os
import openai
import logging


def create_openai_client(api_key=None):
    """
    Create an OpenAI client for API calls.

    Args:
        api_key (str): The OpenAI API key. If not provided, it will be fetched from the environment variable.
    """
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("No OpenAI API key provided or found in environment.")

    client = openai.OpenAI(api_key=api_key)
    logging.info("OpenAI client created successfully.")
    return client


def create_azure_openai_client(api_key=None, endpoint=None):
    """
    Create an Azure OpenAI client for API calls.

    Args:
        api_key (str): The Azure OpenAI API key. If not provided, it will be fetched from the environment variable.
        endpoint (str): The Azure OpenAI endpoint. If not provided, it will be fetched from the environment variable.
    """
    if api_key is None:
        api_key = os.getenv('AZURE_OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("No Azure OpenAI API key provided or found in environment.")

    if endpoint is None:
        endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    if endpoint is None:
        raise ValueError("No Azure OpenAI endpoint provided or found in environment.")

    client = openai.AzureOpenAI(
        api_key=api_key,
        api_version="2025-03-01-preview",
        azure_endpoint=endpoint,
    )
    return client