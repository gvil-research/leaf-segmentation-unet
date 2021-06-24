

def resize2square(img, size, inter_img=cv2.INTER_CUBIC, 
                             inter_mask=cv2.INTER_LINEAR):
  h, w = img.shape[:2]
  c = None if len(img.shape) < 3 else img.shape[2]
  if h == w: 
    if c is None:
      return cv2.resize(img, (size, size), inter_mask)
    else:
      return cv2.resize(img, (size, size), inter_img)
  if h > w: dif = h
  else:     dif = w
  x_pos = int((dif - w)/2.)
  y_pos = int((dif - h)/2.)
  if c is None:
    mask = np.zeros((dif, dif), dtype=img.dtype)
    mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    return cv2.resize(mask, (size, size), inter_mask)
  else:
    mask = np.zeros((dif, dif, c), dtype=img.dtype)
    mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
    return cv2.resize(mask, (size, size), inter_img)