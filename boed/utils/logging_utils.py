"""Simple logging helpers for BOED workflows."""

import os
from datetime import datetime
from typing import Optional

class Logger:
    """Simple logger for BOED workflows."""
    
    def __init__(self, filename: Optional[str] = None, verbose: bool = True):
        """
        Args:
            filename: Log file path (if None, stdout only)
            verbose: Also print to stdout
        """
        self.filename = filename
        self.verbose = verbose
        self.messages: list[str] = []
        
        if filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    def log(self, msg: str, level: str = "INFO") -> None:
        """
        Record a log message.
        
        Args:
            msg: Message text
            level: "INFO", "WARNING", "ERROR"
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] [{level}] {msg}"
        
        self.messages.append(log_msg)
        
        if self.verbose:
            print(log_msg)
        
        if self.filename:
            with open(self.filename, 'a') as f:
                f.write(log_msg + "\n")
    
    def info(self, msg: str) -> None:
        """Log info."""
        self.log(msg, "INFO")
    
    def warning(self, msg: str) -> None:
        """Log warning."""
        self.log(msg, "WARNING")
    
    def error(self, msg: str) -> None:
        """Log error."""
        self.log(msg, "ERROR")
