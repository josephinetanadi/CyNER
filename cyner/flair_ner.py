from flair.data import Sentence
from flair.models import SequenceTagger

from .entity_extraction import EntityExtraction
from .entity import Entity


class Flair(EntityExtraction):
    """
    Entity extraction using Flair NER model
    """
    def __init__(self, config):
        super().__init__(config)
        self.tagger = SequenceTagger.load(config['model'])

    def train(self):
        pass

    def get_entities(self, text):
        sentence = Sentence(text)
        self.tagger.predict(sentence)
        # pred = sentence.to_dict(tag_type='ner')
        entities = []
        for x in sentence.get_spans('ner'):
            # 'labels' are formatted as [(TAG prob), ...]
            entities.append(Entity(x.start_position, x.end_position, x.text, x.get_label("ner").value, x.get_label("ner").score))
        return entities
