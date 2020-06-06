import pandas as pd
import json


def process_record(input_data):
    json_str = json.dumps(input_data)
    df = pd.read_json(json_str, orient='records', lines=True)
    # pd.DataFrame()
    df.drop(columns='ID', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.loc[df.ClaimAmount < 0, 'ClaimAmount'] = 0
    df.Gender, GenderRef = seriesfactorizer(df.Gender)
    df.MariStat, MariStatRef = seriesfactorizer(df.MariStat)
    df['SocioCateg'] = df.SocioCateg.str.slice(0, 4)
    pd.DataFrame(df.SocioCateg.value_counts().sort_values()).rename({'SocioCateg': 'Frequency'}, axis=1)
    df = pd.get_dummies(df, columns=['VehUsage', 'SocioCateg'])
    df = df.select_dtypes(exclude=['object'])
    df['DrivAgeSq'] = df.DrivAge.apply(lambda x: x ** 2)
    df['ClaimsCount'] = df.ClaimInd + df.ClaimNbResp + df.ClaimNbNonResp + df.ClaimNbParking + df.ClaimNbFireTheft + \
                        df.ClaimNbWindscreen
    df.loc[df.ClaimAmount == 0, 'ClaimsCount'] = 0
    df.drop(["ClaimNbResp", "ClaimNbNonResp", "ClaimNbParking", "ClaimNbFireTheft", "ClaimNbWindscreen"],
            axis=1,
            inplace=True)
    return df


def seriesfactorizer(series):
    series, unique = pd.factorize(series)
    reference = {x: i for x, i in enumerate(unique)}
    return series, reference
