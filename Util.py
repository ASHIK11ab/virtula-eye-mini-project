def object_in_walk_frame(leftXBound, rightXBound, mid, imageStartCoord, imageEndCoord):
  xmin, ymin = imageStartCoord
  xmax, ymax = imageEndCoord

  # Object completely inside frame
  if xmin >= leftXBound and xmax <= rightXBound:
    return True

  # Object size is larger than frame.
  if xmin <= leftXBound and xmax >= rightXBound:
    return True

  # Right part of object inside frame.
  if xmin < leftXBound and xmax >= mid - 50:
    return True
  
  # Left part of object inside frame.
  if xmax > rightXBound and xmin <= mid + 50:
    return True

  return False