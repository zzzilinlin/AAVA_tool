#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility to filter rows in a DataFrame by presence of identity and negative words.

Provides `find_text_in_df(df, column='full_text')` which returns a filtered DataFrame.
"""

import re
from tqdm import tqdm

# keyword lists
ethnicity = [
    'indonesiër', 'indonesiërs',
    'afghaan', 'afghanen',
    'allochtoon', 'allochtonen',
    'antilliaan', 'antillianen',
    'arabier', 'arabieren',
    'buitenlander', 'buitenlanders',
    'expat', 'expats',
    'immigrant', 'immigranten',
    'irakees', 'irakezen',
    'marokkaan', 'marokkanen',
    'migrant', 'migranten',
    'moslim', 'moslims',
    'somalier', 'somaliers',
    'surinamer', 'surinamers',
    'syrier', 'syriers',
    'turk', 'turken',
    'vluchteling', 'vluchtelingen',
    'asielzoeker', 'asielbeleid',
    'vreemdeling', 'islamkritiek',
    'migratieachtergrond',
    'arbeidsmigrant', 'arbeidsmigranten',
    'xenofobie', 'xenofoob',
    'kolonialisme', 'koloniaal',
    'diversiteit', 'vooroordelen',
    'probleemwijk', 'asielplaag', 'asieltsunami',
    'gelukzoeker', 'gelukszoeker', 'kanslozen',
    'vluchtelingenstroom',
    'economische migrant', 'transmigrant', 'kennismigrant',
    'blank',
    'migratiestandpunten', 'islamstandpunten'
]

religion = [
    'moslim', 'moslima',
    'islamiet', 'islamieten', 'islam', 'koran',
    'christen', 'christenen', 'christendom',
    'jood', 'joden', 'joods',
    'hindoe', 'hindoes', 'hindoeïsme',
    'boeddhist', 'boeddhisten', 'boeddhisme',
    'atheïst', 'atheïsten',
    'ongelovige', 'ongelovigen',
    'geloofsgemeenschap',
    'religie', 'religieus',
    'islamofobie', 'antisemitisme',
    'thora', 'tora', 'torah',
    'bijbel', 'gelovigen',
    'godslastering', 'blasfemie',
    'marxisme', 'conservatief',
    'cancelcultuur', 'cancelen',
    'mocro', 'mocro maffia',
    'wappie', 'tokkie',
    'domrechts', 'complotdenkers',
    'cultuurmarxisme', 'alt-right',
    'woke', 'wokisme',
    'haatzaaier', 'haatzaaien'
]

high_threat = [
    'afperser', 'agent', 'agente',
    'arrestant', 'arrestanten',
    'autodief', 'autokraker',
    'bajesklant',
    'bandiet', 'bandieten',
    'bankovervaller', 'bankrover',
    'bedelaar', 'bedreiger',
    'bende', 'bendeleden', 'bendeleider', 'bendelid', 'benden', 'bendes',
    'beroepscrimineel',
    'berovingen', 'beschieting',
    'beul', 'boef',
    'bolletjesslikker', 'bolletjesslikkers',
    'bommenmaker', 'bordelen',
    'brandstichter', 'brandstichters',
    'corrupt', 'criminaliteit',
    'crimineel', 'criminelen',
    'cyberpesten',
    'dader', 'daders',
    'delinquent', 'delinquenten',
    'dief', 'draaideurcrimineel',
    'drugsbaas', 'drugsbaron',
    'drugsbende', 'drugsbendes',
    'drugscrimineel', 'drugsdealer', 'drugsdealers',
    'drugsgebruikers',
    'drugshandelaar', 'drugshandelaars',
    'drugssmokkelaar',
    'dubbelagent',
    'fietsendief',
    'gangster', 'gangsterbende',
    'gedetineerde', 'gedetineerden',
    'gegijzelden',
    'gevangenbewaarders',
    'gevangene', 'gevangenen',
    'gevangenisbewaarder', 'gevangenissen',
    'geweldsman',
    'gijzelaar', 'gijzelaars',
    'gijzelnemer', 'gijzelnemers',
    'handlanger',
    'hardrijder',
    'hoofdagent', 'hoofdagente',
    'hoofddader', 'hoofdverdachte',
    'hooligan', 'hooligans',
    'huurmoord', 'huurmoordenaar',
    'illegalen', 'illegaal', 'illegale',
    'inbreker', 'indringer',
    'jeugdbende', 'jeugdbendes', 'jeugddelinquent',
    'kaper', 'kapers',
    'kidnapper', 'kidnappers',
    'kinderlokker', 'kindermisbruiker',
    'kindermoordenaar',
    'kindslaven',
    'krijgsgevangenen',
    'kruimeldief', 'kunstdief',
    'ladykiller',
    'lastpak', 'lastpost',
    'liquidatie',
    'loverboy', 'lovergirls',
    'lustmoordenaar',
    'maffia', 'maffiabaas', 'maffiosi', 'maffioso',
    'massamoordenaar', 'massamoordenaars',
    'mededader', 'medegedetineerde',
    'medegevangene', 'medeplichtige',
    'medeverdachte',
    'mensenhandelaren',
    'mensensmokkelaar', 'mensensmokkelaars',
    'messentrekker',
    'misdaden', 'misdadig',
    'misdadiger', 'misdadigers',
    'misdadigerwapenhandelaar',
    'moordenaar', 'moordenaars',
    'moordmachine', 'moordverdachte',
    'motoragent',
    'neerstak', 'neersteken',
    'ontvoeringen',
    'oplichter', 'overvaller',
    'pedofiel', 'pedofielen',
    'piraten',
    'plunderaar', 'plunderaars',
    'politieagent', 'politieagente', 'politieagenten',
    'politiecommandant', 'politiegeneraal',
    'politiegewonde', 'politieman', 'politiemannen',
    'politiemensen', 'politieofficier',
    'politiepost', 'politierechercheur',
    'politiestaat', 'politievrouw',
    'pyromaan', 'recidivist',
    'relschopper', 'relschoppers',
    'roofmoord', 'roofoverval',
    'scherpschutter', 'schutter',
    'seriemoordenaar',
    'skinheads',
    'slaaf', 'slachtoffers', 'slaven',
    'sluipschutter', 'sluipschutters',
    'smokkelaar', 'smokkelaars',
    'snelheidsduivel',
    'souteneur', 'stalker',
    'straatbende',
    'strafbaar', 'strafklacht',
    'struikrover',
    'tasjesdief',
    'terreurgroep', 'terreurverdachte',
    'terrorist',
    'uitbuiting',
    'vechtersbaas',
    'veelpleger', 'veelplegers',
    'veiligheidsagent', 'veiligheidsagenten',
    'veiligheidspolitie',
    'verdachte', 'verkrachter',
    'vermisten', 'voortvluchtige',
    'vreemdeling',
    'vrouwenhandelaar',
    'wapenhandelaar',
    'winkeldief', 'winkeldievegge',
    'wreker', 'wurgmoord',
    'zakkenrollers',
    'zedendelinquent', 'zedendeliquent',
    'zelfmoordenaar',
    'zwartrijder',
    'probleemwijk', 'probleemjongeren',
    'femicide', 'veiligheid'
]

low_status = [
    'achterlijk', 'achterlijke',
    'achterstanden', 'achterstandskinderen',
    'achterstandsleerling', 'achterstandsleerlingen',
    'achterstandswijken', 'achterstandwijken',
    'achterstelling',
    'alcoholicus', 'alcoholist', 'alcoholiste', 'alcoholisten',
    'analfabeet', 'analfabete', 'analfabeten',
    'armoedig',
    'barbaars',
    'bastaardzoon',
    'bedelaar', 'bedelaars',
    'bijstandsgerechtigden', 'bijstandsgerechtigen',
    'boerenlul',
    'dakloze', 'daklozen',
    'dronkelap',
    'drugsgebruiker', 'drugsgebruikers',
    'drugsrunners', 'drugstoeristen',
    'drugsverslaafde', 'drugsverslaafden',
    'hangjongere', 'hangjongeren',
    'hoer', 'hoerenlopers',
    'hulpbehoevend', 'hulpbehoevende',
    'idioot',
    'junk', 'junkers', 'junks',
    'kansarme', 'kansarmen',
    'kindertehuizen',
    'krottenwijk',
    'laaggeschoolde', 'laaggeschoolden',
    'laagopgeleide', 'laagopgeleiden',
    'loser',
    'malloot',
    'minderwaardig',
    'nestbevuiler',
    'nietsnut',
    'onderklasse',
    'onderontwikkeld',
    'ongeletterde', 'ongeschoolde',
    'overlastgevende',
    'pooier',
    'primitief',
    'probleemjongeren',
    'prostituee', 'prostituees',
    'prostitutiebedrijven',
    'reljongeren',
    'schoolverlaters',
    'slet',
    'sloeber', 'sloebers',
    'spijbelaar', 'spijbelen',
    'straatarm', 'straatkinderen', 'straatprostitutie',
    'sukkel',
    'taalachterstand',
    'tienermoeder', 'tienermoeders',
    'uitkeringgerechtigden', 'uitkeringsgerechtigden',
    'uitwas',
    'verschoppelingen',
    'verslaafde', 'verslaafden',
    'weeskinderen',
    'werkloos', 'werkloze', 'werklozen',
    'werkschuwe',
    'zwerver', 'zwervers'
]

gender = [
    'transgender', 'lgbtq', 'lgbtq+', 'genderideologie',
    'lhbti', 'lhbtqia+',
    'queer', 'non-binair', 'non-binaire',
    'genderneutraal', 'genderdivers',
    'geslacht', 'homo', 'homos', 'lesbienne',
    'travestiet', 'transpersoon',
    'heteronormatief', 'sekswerker',
    'femicide', 'gendergerelateerd geweld',
    'transfobie', 'homofobie', 'victim blaming',
    'aanranding', 'seksuele intimidatie', 'intimidatie',
    'lustmoord', 'gezinsdrama', 'passionele moord', 'passiemoord',
    'gay bashing', 'genderwaanzin',
    'prostituee', 'hoer',
    'woke', 'wokisme',
    'billentikker', 'seksrelatie', 'seksuele relatie'
]

# compile regex patterns (whole word, case insensitive)
ethnicity_pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, ethnicity)) + r')\b', flags=re.IGNORECASE)
#religion_pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, religion)) + r')\b', flags=re.IGNORECASE)
#gender_pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, gender)) + r')\b', flags=re.IGNORECASE)
high_threat_pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, high_threat)) + r')\b', flags=re.IGNORECASE)
low_status_pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, low_status)) + r')\b', flags=re.IGNORECASE)


def check_conditions(text):
    if not isinstance(text, str):
        return False
    
    # count matches
    ethnicity_count = len(ethnicity_pattern.findall(text))
    high_threat_count = len(high_threat_pattern.findall(text))
    low_status_count = len(low_status_pattern.findall(text))
    
    # must have ≥3 ethnicity words
    # and ≥3 total of high_threat + low_status
    return ethnicity_count >= 3 and (high_threat_count + low_status_count) >= 3


def find_text_in_df(df, column: str = 'full_text'):
    """Filter `df` by applying the text check to `column` and return filtered DataFrame.

    Parameters
    - df: pandas.DataFrame
    - column: name of the text column to inspect (default: 'full_text')

    Returns
    - pandas.DataFrame: subset of rows meeting the conditions
    """
    try:
        import pandas as pd
    except Exception:  # pragma: no cover - unlikely
        raise ImportError("pandas is required to use find_text_in_df")

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    cleaned_df = df.dropna(subset=[column])

    # enable tqdm pandas integration and apply the check
    tqdm.pandas(desc="Filtering rows")
    return cleaned_df[cleaned_df[column].progress_apply(check_conditions)]


if __name__ == '__main__':
    # simple CLI: read CSV and print number of filtered rows
    import sys
    import pandas as pd

    if len(sys.argv) < 2:
        print("Usage: python scripts/find_text.py <input_csv> [column]")
        sys.exit(1)

    input_csv = sys.argv[1]
    column = sys.argv[2] if len(sys.argv) > 2 else 'full_text'
    df = pd.read_csv(input_csv)
    res = find_text_in_df(df, column=column)
    print(f"Filtered rows: {len(res)}")





