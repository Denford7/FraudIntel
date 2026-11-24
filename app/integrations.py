
class FintechAdapter:
    """
    Maps a fintech's field names to FraudIntel's
    canonical model feature names.
    """

    def __init__(
        self,
        field_mapping: dict[str, str],
        required_features: list[str],
    ):
        self.field_mapping = field_mapping
        self.required_features = required_features

        invalid_targets = [
            target
            for target in field_mapping.values()
            if target not in required_features
        ]

        if invalid_targets:
            raise ValueError(
                f"Unknown FraudIntel features: {invalid_targets}"
            )


    def transform_transaction(
        self,
        transaction: dict,
    ) -> dict:
        """Convert fintech data into FraudIntel's canonical format."""

        mapped_transaction = {}

        for fintech_field, value in transaction.items():

            # Use configured mapping when one exists
            fraudintel_field = self.field_mapping.get(
                fintech_field,
                fintech_field
            )

            # Keep only recognised model features
            if fraudintel_field in self.required_features:
                mapped_transaction[fraudintel_field] = value

        missing_features = [
            feature
            for feature in self.required_features
            if feature not in mapped_transaction
        ]

        if missing_features:
            raise ValueError(
                f"Missing {len(missing_features)} required model features: "
                f"{missing_features[:10]}"
            )

        return mapped_transaction

