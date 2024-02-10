from cogelot.structures.vima import VIMAInstance


class VIMAInstanceTransform:
    """Transform VIMA instances by applying a function to them.

    This will allow us to, especially during evaluation, to modify the instance before we actually
    provide it to the environment.
    """

    def __call__(self, instance: VIMAInstance) -> VIMAInstance:
        """Return the instance without transforming it.."""
        raise NotImplementedError


class NoopTransform(VIMAInstanceTransform):
    """Do not transform the instance."""

    def __call__(self, instance: VIMAInstance) -> VIMAInstance:
        """Return the instance without transforming it.."""
        return instance


class ChainTransform(VIMAInstanceTransform):
    """Do GobbledygookWord on top of the textual transform."""

    def __init__(self, *transforms: VIMAInstanceTransform) -> None:
        self._transforms = transforms

    def __call__(self, instance: VIMAInstance) -> VIMAInstance:
        """Process the instance."""
        for transform in self._transforms:
            instance = transform(instance)
        return instance
