from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BaseModel(ABC):
    
    @abstractmethod
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        pass
    
    @abstractmethod
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def load_model(self):
        pass 