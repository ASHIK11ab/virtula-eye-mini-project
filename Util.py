def object_in_walk_frame(leftXBound, rightXBount, imageStartCoord, imageEndCoord):
  xmin, ymin = imageStartCoord
  xmax, ymax = imageEndCoord

  if xmin >= leftXBound and xmax <= rightXBount:
    return True

  if xmin <= leftXBound and xmax >= rightXBount:
    return True

  return False