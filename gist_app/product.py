from PIL import Image


# A class to hold a product
class Product:

    # We'll have an image, label, and category
    def __init__(self, image: Image, label: str, category: str, url: str = None) -> None:
        self._image = image
        self._label = label
        self._category = category
        self._url = url

    @property
    def url(self) -> str:
        return self._url
    
    @url.setter
    def url(self, url: str) -> None:
        self._url = url

    @property
    def image(self) -> Image:
        return self._image
    
    @image.setter
    def image(self, image: Image) -> None:
        self._image = image

    @property
    def label(self) -> str:
        return self._label
    
    @label.setter
    def label(self, label: str) -> None:
        self._label = label

    @property
    def category(self) -> str:
        return self._category
    
    @category.setter
    def category(self, category: str) -> None:
        self._category = category
