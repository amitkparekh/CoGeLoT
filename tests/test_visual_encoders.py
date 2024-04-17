from pytest_cases import parametrize_with_cases

from cogelot.nn.visual_encoders import (
    ObjectCentricVisualEncoder,
    PatchesVisualEncoder,
    TheirEncoder,
    VisualEncoder,
)
from vima.nn.obj_encoder import GatoMultiViewRGBEncoder, ObjEncoder


class TheirObjEncoderCases:
    def case_obj_encoder(self, embed_dim: int) -> tuple[ObjEncoder, type[VisualEncoder]]:
        encoder = ObjEncoder(
            transformer_emb_dim=embed_dim,
            views=["front", "top"],
            vit_output_dim=embed_dim,
            vit_resolution=32,
            vit_patch_size=16,
            vit_width=embed_dim,
            vit_layers=2,
            vit_heads=2,
            bbox_mlp_hidden_dim=embed_dim,
            bbox_mlp_hidden_depth=2,
        )
        return encoder, ObjectCentricVisualEncoder

    def case_gato_obj_encoder(
        self, embed_dim: int
    ) -> tuple[GatoMultiViewRGBEncoder, type[VisualEncoder]]:
        encoder = GatoMultiViewRGBEncoder(
            emb_dim=embed_dim,
            views=["front", "top"],
            img_size=(64, 128),
            vit_patch_size=32,
            vit_width=embed_dim,
            vit_layers=2,
            vit_heads=2,
        )
        return encoder, PatchesVisualEncoder


@parametrize_with_cases("obj_encoder,encoder_class", cases=TheirObjEncoderCases)
def test_create_obj_encoder_from_their_encoder(
    obj_encoder: TheirEncoder, encoder_class: type[VisualEncoder]
) -> None:
    converted_class = encoder_class.from_their_encoder(obj_encoder)
    assert isinstance(converted_class, encoder_class)
