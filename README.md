## Implementation of algorithm from the paper about keystroke dynamics spoofing

![Bounding Box Detection](.github/boxes.gif "Bounding Box Detection")

Algorithm is implemented based on this paper (DOI: 10.1186/s40537-022-00662-8):
> Spoofing keystroke dynamics authentication
> through synthetic typing pattern extracted
> from screen‑recorded video

Since I don't know what dataset was used to train the model from the paper
this repo uses [The Chars74K dataset](https://info-ee.surrey.ac.uk/CVSSP/demos/chars74k/) instead.

## Usage:
```
poetry run kdfv --help
```

### Example:

To save images of all separated characters with their convexity (Figure 7 in the paper) use:
```
poetry run kdfv kunit --convexity <video file> <destination>
```

![Convex Defect](.github/convex.gif "Convex Defect")

THIS IS CURRENTLY WIP. Higher haracter separation precision is necessary for it to have a chance to work.
