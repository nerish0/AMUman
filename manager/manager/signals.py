import logging

from constance.signals import config_updated
from django.dispatch import receiver

from manager.components.scheduler import ThreadedScheduler

from .components.queueManager import QueueManager

log = logging.getLogger("rich")
