<?php

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $categoryName = test_input($_POST["categoryName"]);

    $servername = "localhost";
    $user = "root";
    $password = "";
    $dbname = "final_project";

// Create connection
    $conn = new mysqli($servername, $user, $password, $dbname);
    mysqli_set_charset($conn, "utf8");
// Check connection
    if ($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error);
    }

    $counter = 0;
    $innerBody = '';
    $exist = false;
    $sql = "SELECT id,name,price,image_dir FROM good where categoryName = \"" . $categoryName . "\"";
    $result = $conn->query($sql);

    if ($result->num_rows > 0) {
        // output data of each row
        while ($row = $result->fetch_assoc()) {
//            $row["email"]
            if ($counter % 3 == 0) {
                $innerBody .= '<div class="row">';
            }
            $counter++;
            $innerBody .= '<div class="column" onclick="formPost(' . $row["id"] . ')">
        <div class="image">
            <img src="' . $row["image_dir"] . '">
        </div>
        <br>
        نام محصول:
        ' . $row["name"] . '
        <br>
        <br>
        قیمت محصول:
        ' . $row["price"] . '
        تومان
        <br>
        <br>
    </div>';
            if ($counter % 3 == 0) {
                $innerBody .= '</div>';
            }
        }
    }
    if ($counter % 3 != 0) {
        $innerBody .= '</div>';
    }


}

$string = '<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>محصولات بر اساس دسته بندی</title>
    <link rel="stylesheet" href="../css/goodByCategory.css">
</head>
<body>
<form method="post" action="../php/buyGood.php" id="goodByCategoryForm" name="goodByCategoryForm">
</form>
'.$innerBody.'

<script src="../js/goodByCategory.js"></script>
</body>
</html>';

echo $string;

function test_input($data)
{
    $data = trim($data);
    $data = stripslashes($data);
    $data = htmlspecialchars($data);
    return $data;
}

?>