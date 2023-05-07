# Imports 1


# Variables


# Functions and classes
def _adf_exists_check(adf: str, download_url: str) -> None:
    """
    Checks that an adf was downloaded.

    :param (str) adf:
        The adf file downloaded from OpenADAS. Types that can be checked are adf11 and adf15.
    :param (str) download_url:
        OpenADAS URL to download the adf

    :return: None

    :raise ValueError:
        No adf downloaded.
    """
    if 'OPEN-ADAS Error' in adf:
        # Error message template
        err_msg: str = "No ADF15 was downloaded, '{adf_name}' does not exist (url: '{download_url}')!"
        # Extract file name from url based on adf type
        adf_type: str = download_url.split('download')[1].split('/')[1]
        if adf_type == 'adf11':
            adf_name: str = download_url.split('download')[1].split('/')[-1]
        elif adf_type == 'adf15':
            adf_name: str = '#'.join(download_url.split('][')[1:]).split('/')[1]
        else:
            _err_msg: str = f"'{adf_type}' is not an acceptable type, must be an adf11 or adf15!"
            raise NotImplementedError(_err_msg)
        # Write error message and raise error
        err_msg = err_msg.format(adf_name=adf_name, download_url=download_url)
        raise ValueError(err_msg)


def main() -> None:
    pass


if __name__=="__main__":
    main()
    