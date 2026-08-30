from traffic_vision.ai_labelers import (
    GroundingDinoProposalProvider,
    ProposedImage,
    ProposalProvider,
    YoloWorldProposalProvider,
)


def test_proposal_adapters_are_importable_without_loading_models() -> None:
    assert ProposedImage.__name__ == "ProposedImage"
    assert ProposalProvider.__name__ == "ProposalProvider"
    assert YoloWorldProposalProvider.DEFAULT_PROMPTS[0] == "toy car"
    assert GroundingDinoProposalProvider.__name__ == "GroundingDinoProposalProvider"
