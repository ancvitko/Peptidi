import numpy as np
import pandas as pd
import peptides

class PeptidePropertiesEncoder:
    def encode(self, sequences):
        groups = (
            ('A', 'C', 'G', 'S', 'T'),                                  # Tiny
            ('A', 'C', 'D', 'G', 'N', 'P', 'S', 'T', 'V'),              # Small 
            ('A', 'I', 'L', 'V'),                                       # Aliphatic
            ('F', 'H', 'W', 'Y'),                                       # Aromatic
            ('A', 'C', 'F', 'G', 'I', 'L', 'M', 'P', 'V', 'W', 'Y'),    # Non-polar
            ('D', 'E', 'H', 'K', 'N', 'Q', 'R', 'S', 'T'),              # Polar
            ('D', 'E', 'H', 'K', 'R'),                                  # Charged
            ('H', 'K', 'R'),                                            # Basic
            ('D', 'E')                                                  # Acidic
        )

        X = []
        for sequence in sequences:
            sequence = sequence.upper()

            peptide = peptides.Peptide(sequence)
            x = [
                peptide.cruciani_properties()[0],
                peptide.cruciani_properties()[1],
                peptide.cruciani_properties()[2],
                peptide.instability_index(),
                peptide.boman(),
                peptide.hydrophobicity("Eisenberg"),
                peptide.hydrophobic_moment(angle=100, window=min(len(sequence), 11)),
                peptide.aliphatic_index(),
                peptide.isoelectric_point("Lehninger"),
                peptide.charge(pH=7.4, pKscale="Lehninger"),
            ]

            # Count tiny, small, aliphatic, ..., basic and acidic amino acids
            for group in groups:
                count = 0
                for amino in group:
                    count += sequence.count(amino)
                # x.append(count)
                x.append(count / len(sequence))
            X.append(x)
        return np.array(X)

def process_file(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)
    sequences = df['sequence'].tolist()

    # Instantiate the encoder
    properties_encoder = PeptidePropertiesEncoder()

    # Encode the sequences
    encoded_properties = properties_encoder.encode(sequences)

    # Define the header
    header = [
        "Cruciani_1", "Cruciani_2", "Cruciani_3", "Instability_Index", "Boman_Index",
        "Hydrophobicity_Eisenberg", "Hydrophobic_Moment", "Aliphatic_Index", 
        "Isoelectric_Point_Lehninger", "Charge_pH7.4_Lehninger"
    ]

    groups = [
        "Tiny", "Small", "Aliphatic", "Aromatic", "Non_polar", "Polar", "Charged", "Basic", "Acidic"
    ]

    for group in groups:
        header.append(f"Freq_{group}")

    # Save the encoded properties to a CSV file with the header
    with open(output_file, 'w') as f:
        # Write the header
        f.write(','.join(header) + '\n')

        # Write the data
        np.savetxt(f, encoded_properties, delimiter=',', fmt='%f')

if __name__ == "__main__":
    process_file('peptides_label_0.csv', 'encoded_peptides_label_0.csv')
    process_file('peptides_label_1.csv', 'encoded_peptides_label_1.csv')

    print("Encoded files created successfully.")
