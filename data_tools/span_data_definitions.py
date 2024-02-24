NONE_LABEL = 'X' # must not be equal to any label abbreviation in label definition dict

# legacy label definitions, and abbreviations
LABEL_DEF = {
    'OBJETIVOS': 'O', 'AGENTE': 'A', 'FACILITADORES': 'F',
    'PARTIDARIOS': 'P', 'VÍCTIMAS': 'V', 'EFECTOS_NEGATIVOS': 'E',
}

# maping from .json annotation label abbreviations to the official label names
SPAN_LABELS_OFFICIAL = {
    'O': 'OBJECTIVE', 'A': 'AGENT', 'F': 'FACILITATOR',
    'P': 'CAMPAIGNER', 'V': 'VICTIM', 'E': 'NEGATIVE_EFFECT',
}
