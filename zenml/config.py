"""
Configuration pour envoi de message dans Slack
"""
import os

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/xxxx/yyyy")
