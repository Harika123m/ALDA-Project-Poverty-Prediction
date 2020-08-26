import pandas as pd
import numpy as np

# setting warnings off
pd.options.mode.chained_assignment = None

# Rename column names
columns = {
    'v2a1': 'MonthlyRent',
    'hacdor': 'OverCrBedroom',
    'hacapo': 'OverCrRooms',
    'idhogar': 'HouseholdID',
    'v14a': 'Bathroom',
    'v18q1': 'NumOfTablets',
    'r4h3': 'NumMales',
    'r4m3': 'Numfemales',
    'r4t3': 'HHSize',
    'edjefe': 'MaleEdu',
    'edjefa': 'FemaleEdu',
    'hogar_adul': 'NumAdults',
    'escolari': 'Educ',
    'rez_esc': 'EduYearsRem',
    'r4t1': 'NumYoung12',
    'r4t2': 'NumElder12',
    'pisonotiene': 'NoFLoor',
    'sanitario1': 'Toilet',
    'energcocinar1': 'Energy',
    'cielorazo': 'Ceiling',
    'abastaguano': 'Water',
    'paredblolad': 'wall_Block/Brick',
    'paredzocalo': 'wall_socket',
    'paredpreb': 'wall_Fab/Cement',
    'pareddes': 'wall_Wastematerial',
    'paredmad': 'wall_wood',
    'paredzinc': 'wall_zinc',
    'paredfibras': 'wall_NatFiber',
    'paredother': 'wall_other',
    'pisomoscer': 'Floor_mosaic/ceramic/terrazo',
    'pisocemento': 'Floor_cement',
    'pisoother': 'Floor_other',
    'pisonatur': 'Floor_Natural',
    'pisomadera': 'Floor_wood',
    'techozinc': 'Roof_foil/zinc',
    'techoentrepiso': 'Roof_fiber/cement/mezzanine',
    'techocane': 'Roof_NatFibers',
    'techootro': 'Roof_other',
    'abastaguadentro': 'Water_InsideDwelling',
    'abastaguafuera': 'Water_OutsideDwelling',
    'sanitario2': 'Toilet_Sewer/Cesspool',
    'sanitario3': 'Toilet_SepticTank',
    'sanitario5': 'Toilet_BlackHole/Letrine',
    'sanitario6': 'Toilet_other',
    'energcocinar2': 'Cooking_electricity',
    'energcocinar3': 'Cooking_gas',
    'energcocinar4': 'Cooking_charcoal',
    'elimbasu1': 'Disposal_TankerTruck',
    'elimbasu2': 'Disposal_BotanHollow/Buried',
    'elimbasu3': 'Disposal_burning',
    'elimbasu4': 'Disposal_ThrowingUnoccu',
    'elimbasu5': 'Disposal_ThrowingWater',
    'elimbasu6': 'Disposal_other',
    'epared1': 'Wall_Replace_bad',
    'epared2': 'Wall_Replace_regular',
    'epared3': 'Wall_Replace_good',
    'etecho1': 'Roof_replace_bad',
    'etecho2': 'Roof_replace_regular',
    'etecho3': 'Roof_replace_good',
    'eviv1': 'Floor_Replace_bad',
    'eviv2': 'Floor_Replace_regular',
    'eviv3': 'Floor_Replace_good',
    'tipovivi1': 'Rent_Replace_OwnFullyPaid',
    'tipovivi2': 'Rent_Replace_OenInstallments',
    'tipovivi3': 'Rent_Replace_Rented',
    'tipovivi4': 'Rent_Replace_Precarious',
    'tipovivi5': 'Rent_Replace_others',
    'lugar1': 'Region_Replace_Central',
    'lugar2': 'Region_Replace_Chorotega',
    'lugar3': 'Region_Replace_AfricoCentral',
    'lugar4': 'Region_Replace_Brunca',
    'lugar5': 'Region_Replace_HuetarAtlantica',
    'lugar6': 'Region_Replace_Huetar Norte',
    'instlevel1': 'edu_no',
    'instlevel2': 'edu_primary_incomplete',
    'instlevel3': 'edu_primary_complete',
    'instlevel4': 'edu_secondary_incomplete',
    'instlevel5': 'edu_secondary_complete',
    'instlevel6': 'edu_tech_secondary_incomplete',
    'instlevel7': 'edu_tech_secondary_completeedu_ug',
    'instlevel8': 'edu_pg',
    'public': 'elec_public',
    'planpri': 'elec_planpri',
    'coopele': 'elec_coopele'
}

ind_ordinal = ['Education', 'EduYearsRem', 'Educ', 'age']
ind_bool = ['mobilephone', 'dis','NumOfTablets']
ind_categories = ['Relation', 'Status']


def rename_columns(data):
    data.rename(columns=columns, inplace=True)


def target_plotter(x, data, filename):
    data = data.groupby(['Target',x])[x].count().unstack().fillna(0)
    ax = data.plot.bar(stacked=True)
    fig = ax.get_figure()
    fig.savefig('fig/{}'.format(filename))


def clean_up_data(data_file):
    """This function is divided into two sub parts.
     The first part computes the household characteristics.
     And the next part takes care of individuals characteristics."""

    # read the df as dataframe from .csv files
    df = pd.read_csv(data_file)
    file_name = 'Train'
    if 'test' in data_file:
        print('Cleaning up Testing data')
        df['Target'] = 0  # Adding target for test
        file_name = 'Test'
    else:
        print('Cleaning up Training data')

    print(df.groupby('idhogar'))
    # get the shape
    print("Initial Shape of the df", df.shape)

    # Print the Number of Integer based columns
    print("Number of columns that has integer values : ", df.select_dtypes(np.int64).shape[1])

    # Print the Number of boolean columns
    print("Number of columns that has boolean values : ", df.select_dtypes(np.int64).nunique().value_counts()[2])

    # Print the columns that has non numerical df
    print("Columns that has non numerical values : ", df.select_dtypes('object').columns)
    change_yes_no = {'yes': 1, 'no': 0}
    df['dependency'] = df['dependency'].replace(change_yes_no).astype(np.float64)
    df['edjefa'] = df['edjefa'].replace(change_yes_no).astype(np.float64)
    df['edjefe'] = df['edjefe'].replace(change_yes_no).astype(np.float64)

    print("Unique house heads are ", df['parentesco1'].value_counts()[1])

    # RemoveRepeated(redundant) attributes
    df.drop(['tamhog', 'tamviv', 'hhsize', 'hogar_total', 'r4h1', 'r4h2', 'r4m1', 'r4m2'], axis=1, inplace=True)

    # Remove all the squared attributes (They are for calculation purpose)
    df.drop(
        ['agesq','SQBhogar_total','SQBedjefe','SQBhogar_nin','SQBovercrowding','SQBdependency','SQBmeaned','SQBescolari','SQBage','male','female','v18q'],
        axis=1,
        inplace=True
    )

    # Removin houses without heads as this wont contribute towards the household step in the future
    households_leader = df.groupby('idhogar')['parentesco1'].sum()
    # Find households without a head and removing them
    households_no_head = df.loc[df['idhogar'].isin(households_leader[households_leader == 0].index), :]
    df.drop(households_no_head.index.tolist(), inplace=True, axis=0)
    df.reset_index()

    # Lets categorize all the Binary attributes into a categorical singular column
    categories = {
        'OutsideWall': ['wall_Block/Brick', 'wall_socket', 'wall_Fab/Cement', 'wall_Wastematerial', 'wall_wood', 'wall_zinc', 'wall_NatFiber', 'wall_other'],
        'Floor': ['Floor_mosaic/ceramic/terrazo', 'Floor_cement', 'Floor_other', 'Floor_Natural', 'Floor_wood'],
        'Roof': ['Roof_foil/zinc', 'Roof_fiber/cement/mezzanine', 'Roof_NatFibers', 'Roof_other'],
        'Region': ['Region_Replace_Central', 'Region_Replace_Chorotega', 'Region_Replace_AfricoCentral', 'Region_Replace_Brunca', 'Region_Replace_HuetarAtlantica', 'Region_Replace_Huetar Norte'],
        'SourceOfElec': ['elec_public', 'elec_planpri', 'elec_coopele'],
        'Toilet': ['Toilet_Sewer/Cesspool', 'Toilet_SepticTank', 'Toilet_BlackHole/Letrine', 'Toilet_other'],
        'SOCooking': ['Cooking_electricity', 'Cooking_gas', 'Cooking_charcoal'],
        'RubbishDisposal': ['Disposal_TankerTruck', 'Disposal_BotanHollow/Buried', 'Disposal_burning', 'Disposal_ThrowingUnoccu', 'Disposal_ThrowingWater', 'Disposal_other'],
        'Rent': ['Rent_Replace_OwnFullyPaid', 'Rent_Replace_OenInstallments', 'Rent_Replace_Rented', 'Rent_Replace_Precarious', 'Rent_Replace_others'],
        'Relation': ['parentesco1', 'parentesco2', 'parentesco3', 'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9', 'parentesco10', 'parentesco11', 'parentesco12'],
        'Status': ['estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7'],
        'Water': ['Water_InsideDwelling', 'Water_OutsideDwelling'],
    }

    ordinal_categories = {
        'WallCondition': ['Wall_Replace_bad', 'Wall_Replace_regular', 'Wall_Replace_good'],
        'RoofCondition': ['Roof_replace_bad', 'Roof_replace_regular', 'Roof_replace_good'],
        'FloorCondition': ['Floor_Replace_bad', 'Floor_Replace_regular', 'Floor_Replace_good'],
        'Education': ['edu_no', 'edu_primary_incomplete', 'edu_primary_complete', 'edu_secondary_incomplete',
                      'edu_secondary_complete', 'edu_tech_secondary_incomplete', 'edu_tech_secondary_complete''edu_ug',
                      'edu_pg'],
    }

    # Rename the columns
    rename_columns(df)

    # Make multiple binary variables into 1 categorical variable
    for category, sub_cat in categories.items():
        df[category] = df[sub_cat].idxmax(axis=1)
        # once merged into a single attribute, we can drop the individual attributes
        df.drop(sub_cat, axis=1, inplace=True)

    # We can also ordinalize binary values
    # For example : Wall condition bad,regular, good can be replaced with 0,1,2

    for category, sub_cat in ordinal_categories.items():
        df[category] = np.argmax(np.array(df[sub_cat]), axis=1)
        # once merged into a single attribute, we can drop the individual attributes
        df.drop(sub_cat, axis=1, inplace=True)

    # We can also create a new House condition value based on the roof/wall/floor condition
    df['HouseCondition'] = df['WallCondition'] + df['RoofCondition'] + df['FloorCondition']

    target_plotter('HouseCondition', df, 'HouseCondition_{}.png'.format(file_name))

    # get the shape - this gives the current number of attributes
    print(df.shape)

    # How well the source of electricity contributes to target
    target_plotter('SourceOfElec', df, 'SourceOfElec_{}.png'.format(file_name))

    # To check how well the target alignes with house with wall/ceiling/roof

    # Make the total dependents (people less than 19 years and greater than 65)
    df['dependCitizens'] = df['hogar_nin'] + df['hogar_mayor']

    # Now drop the columns of dependents    since we have total
    df.drop(['hogar_nin', 'hogar_mayor'],axis=1,inplace=True)

    # Getting the number of households which have discrepancies
    # groupby and nunique > 1 returns households which have more than 1 target
    house_hold_targets = df.groupby('HouseholdID')['Target'].apply(lambda x: x.nunique() > 1)
    house_hold_targets = house_hold_targets[house_hold_targets == True]
    print("Number of households having non unique targets : {}".format(len(house_hold_targets)))

    # Setting all the households to have the same target as the house head
    for _house in house_hold_targets.index:
        df.loc[df['HouseholdID'] == _house, 'Target'] = df[(df['HouseholdID'] == _house) & (df['Relation'] == 'parentesco1')]['Target']

    # get the shape of the df after dealing adults df
    print("After deleting the 2 columns and  aggregating them", df.shape)
    df_target = df['Target']

    # Unique HouseHolds
    households = np.unique(df['HouseholdID'])
    # Unique Houses has same ids
    print("Individuals length ", len(df['HouseholdID']))
    print("House holds length ", len(households))

    print("Distribution of 4 classes in Target variable : ")

    print(df_target.value_counts())

    # remove na's
    # Filling Number of Tablets as 0 if nan
    df.fillna(value=0, inplace=True)

    # filling education remaining with the threshold
    df.loc[df['EduYearsRem'] > 5, 'EduYearsRem'] = 5

    # printing columns
    print(df.columns.tolist())
    print('Total number of columns : {}'
          ''.format(len(df.columns.tolist())))

    # Let us aggregate the individual rows in the household level and merge it back to the main
    individual_df = df.copy()
    # The individual df should contain all the records that are correspondin to the individual features
    df.drop(ind_bool+ind_ordinal, axis=1, inplace=True)
    individual_df = individual_df[ind_bool+ind_ordinal+['HouseholdID']]

    # Adding new dimensions for individuals
    individual_df['Educ/age'] = individual_df['Educ'] / individual_df['age']
    individual_df['Education/age'] = individual_df['Education'] / individual_df['age']
    individual_df['tech'] = individual_df['NumOfTablets'] + individual_df['mobilephone']

    # Define custom function
    range_ = lambda x: x.max() - x.min()
    range_.__name__ = 'range_'

    # aggregated_cols = individual_df.groupby('HouseholdID').mean().reset_index()
    aggregated_cols = individual_df.groupby('HouseholdID').agg(['min', 'max', 'sum', 'count', 'std', range_])
    new_col = [
        f'{c}-{stat}'
        for c in aggregated_cols.columns.levels[0]
        for stat in aggregated_cols.columns.levels[1]
    ]

    aggregated_cols.columns = new_col

    corr_matrix = aggregated_cols.corr()

    # Select upper triangle of correlation matrix
    corr_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    aggregated_cols.drop([column for column in corr_matrix.columns if any(abs(corr_matrix[column]) > 0.95)], axis=1, inplace=True)
    aggregated_cols.fillna(0, inplace=True)
    house_hold_df = df[df['Relation'] == 'parentesco1']
    house_hold_df = pd.merge(house_hold_df, aggregated_cols, on='HouseholdID')
    print("Number of Rows in House_Hold_df : {} and Aggregated_Cols : {}".format(house_hold_df.shape[0], aggregated_cols.shape[0]))

    # removing final individual level columns
    house_hold_df.drop(['Id', 'Relation', 'HouseholdID'], axis=1, inplace=True)
    house_hold_df.to_csv('{}HouseHold.csv'.format(file_name), index=False)
    print("Done!")


if __name__ == '__main__':
    clean_up_data('data/train.csv')
    clean_up_data('data/test.csv')
